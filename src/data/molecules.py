import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Iterator, Optional
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import MiniBatchKMeans
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
import random 

# ------------------------------
# utility: balanced split
# ------------------------------
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import MiniBatchKMeans

def balanced_split(
    xs: torch.Tensor,
    ys: torch.Tensor,
    n_support: int,
    *,
    strategy: str | None = "quantile",  # None/"random", "quantile", "kmeans", "quantile+kmeans"
    n_bins: int = 5,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (sup_idx, que_idx) as torch.long CPU tensors.

    xs : [N, D] features (for clustering if needed)
    ys : [N] targets (for quantile binning if needed)
    n_support : number of support samples to choose

    Notes
    -----
    - Falls back to random if stratification is infeasible (too many strata, tiny bins, etc.).
    - Always uses exactly n_support for support and the rest for query.
    """
    assert xs.ndim >= 1 and ys.ndim == 1
    N = int(ys.shape[0])
    if n_support <= 0 or n_support >= N:
        raise ValueError(f"n_support must be in [1, {N-1}], got {n_support}")

    def _random_split() -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(random_state)
        sup_idx = rng.choice(N, size=n_support, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[sup_idx] = True
        que_idx = np.where(~mask)[0]
        return (torch.as_tensor(sup_idx, dtype=torch.long),
                torch.as_tensor(que_idx, dtype=torch.long))

    # Fast path: pure random split
    if strategy is None or str(strategy).lower() == "random":
        return _random_split()

    strat_labels = None
    s = str(strategy).lower()

    # Quantile bins from targets
    bins = None
    if "quantile" in s:
        y = ys.detach().cpu().numpy().astype(np.float64).ravel()
        edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)  # collapse duplicated edges
        if len(edges) > 2:
            bins = np.digitize(y, edges[1:-1], right=True)  # 0..(#bins-1)

    # KMeans clusters from xs
    clusters = None
    if "kmeans" in s:
        if n_clusters is not None and n_clusters < N:
            km = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,  # safer across sklearn versions
            )
            clusters = km.fit_predict(xs.detach().cpu().numpy())

    # Combine labels
    if bins is not None and clusters is not None:
        strat_labels = bins * (clusters.max() + 1) + clusters
    elif bins is not None:
        strat_labels = bins
    elif clusters is not None:
        strat_labels = clusters
    else:
        strat_labels = None

    # Validate strata
    if strat_labels is not None:
        uniq, counts = np.unique(strat_labels, return_counts=True)
        # need at least 2 per stratum (so both support and query can be non-empty)
        if (counts < 2).any():
            strat_labels = None
        # also require we can place >=1 per class into support
        elif len(uniq) > n_support:
            strat_labels = None

    if strat_labels is None:
        return _random_split()

    # Stratified split (guard for rare errors)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_support, random_state=random_state)
        sup_idx, que_idx = next(sss.split(np.zeros((N, 1)), strat_labels))
        return (torch.as_tensor(sup_idx, dtype=torch.long),
                torch.as_tensor(que_idx, dtype=torch.long))
    except Exception:
        # As a last resort, never crash the loader
        return _random_split()


def _seed_worker(worker_id: int):
    # torch sets per-worker seed; derive numpy/python seeds from it
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MolecularTasks(Dataset):
    def __init__(self,
                 path: str | Path | None = None,
                 dataf: pd.DataFrame | None = None,
                 transform=None,
                 target_transform=None,
                 assignTasks=False,
                 bits=2048,
                 radius=2,
                 featurization: str = "FP"):
        self.transform = transform
        self.target_transform = target_transform
        self.bits = bits
        self.radius = radius
        self.featurization = featurization  # "FP" or "GNN"

        # RDKit descriptor list (filtered)
        self.descriptors = [(name, func)
                            for name, func in Descriptors.descList
                            if not name.startswith(("fr_", "Ipc"))]

        if dataf is not None:
            raw_df = dataf.copy()
        elif path is not None:
            path = Path(path)
            if path.suffix == ".xlsx":
                raw_df = pd.read_excel(path)
            elif path.suffix == ".csv":
                raw_df = pd.read_csv(path)
            else:
                raise ValueError(f"Unknown file extension {path.suffix!r}; expected .xlsx or .csv")
        else:
            raise ValueError("Provide either `dataf` or `path`.")

        # Expect a 'task_id' column
        self.tasks = dict(tuple(raw_df.groupby("task_id")))
        self.task_keys = sorted(self.tasks.keys())

    def __len__(self):
        return len(self.task_keys)

    def __getitem__(self, index):
        task_key = self.task_keys[index]
        task_df = self.tasks[task_key].reset_index(drop=True)
        # NOTE: one MoleculeFormTask == one meta-task (iterable over samples)
        return MoleculeFormTask(task_key,
                                task_df,
                                descriptors=self.descriptors,
                                transform=self.transform,
                                target_transform=self.target_transform,
                                bits=self.bits,
                                radius=self.radius,
                                featurization=self.featurization)

# ---------------------------------
# Single Task Dataset (per meta-task)
# ---------------------------------
class MoleculeFormTask(Dataset):
    FORM_COLS = [
        "Cationic_Lipid_to_mRNA_weight_ratio",
        "Cationic_Lipid_Mol_Ratio",
        "Phospholipid_Mol_Ratio",
        "Cholesterol_Mol_Ratio",
        "PEG_Lipid_Mol_Ratio",
    ]

    def __init__(self, task_key, task_df, descriptors,
                 transform=None, target_transform=None,
                 bits=2048, radius=2, featurization="FP"):
        self.task_key = task_key
        self.task_df = task_df
        self.transform = transform
        self.target_transform = target_transform
        self.bits = bits
        self.radius = radius
        self.descriptors = descriptors
        self.featurization = featurization

        if self.featurization not in ("FP", "GNN"):
            raise ValueError("featurization must be 'FP' or 'GNN'")

        self._build()

    def _build(self):
        df = self.task_df.dropna(subset=["SMILES", "TARGET", *self.FORM_COLS])

        self._targets = []
        if self.featurization == "FP":
            # build Morgan+formulation tensor
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius, fpSize=self.bits, countSimulation=False
            )
            feats = []
            for _, row in df.iterrows():
                smi = str(row["SMILES"])
                form = row[self.FORM_COLS].values.astype(np.float32)
                mol = Chem.MolFromSmiles(smi, sanitize=True)
                if mol is None:
                    fp = np.zeros((self.bits,), dtype=np.float32)
                else:
                    fp = gen.GetFingerprint(mol)
                full = np.concatenate([fp, form]).astype(np.float32)
                feats.append(full)
                self._targets.append(np.float32(row["TARGET"]))
            self._X = torch.tensor(np.stack(feats), dtype=torch.float32)
            self._smiles = None
            self._X_d = None

        else:  # "GNN": pass SMILES + X_d (formulation) separately
            smiles = []
            x_d = []
            for _, row in df.iterrows():
                smi = str(row["SMILES"])
                form = row[self.FORM_COLS].values.astype(np.float32)
                smiles.append(smi)
                x_d.append(form)
                self._targets.append(np.float32(row["TARGET"]))
            self._smiles = smiles
            self._X_d = torch.tensor(np.stack(x_d), dtype=torch.float32)
            self._X = None

        self._targets = torch.tensor(self._targets, dtype=torch.float32)

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, idx):
        # Return (x, y) in FP mode; (smiles, X_d, y) in GNN mode
        y = self._targets[idx]
        if self.featurization == "FP":
            x = self._X[idx]
            if self.transform is not None:
                x = self.transform(x)
            if self.target_transform is not None:
                y = self.target_transform(y)
            return x, y
        else:
            smi = self._smiles[idx]
            x_d = self._X_d[idx]
            if self.transform is not None:
                x_d = self.transform(x_d)
            if self.target_transform is not None:
                y = self.target_transform(y)
            return smi, x_d, y

# ----------------------------
# Collate fns
# ----------------------------
def _check_indices(sup_idx: torch.Tensor, que_idx: torch.Tensor, N: int, task_key):
    # flatten & move to CPU numpy
    sup = np.asarray(sup_idx.detach().cpu(), dtype=np.int64).ravel()
    que = np.asarray(que_idx.detach().cpu(), dtype=np.int64).ravel()

    # 1) no overlap
    inter = np.intersect1d(sup, que, assume_unique=False)
    if inter.size > 0:
        raise RuntimeError(
            f"[{task_key}] Support/query overlap: {inter.tolist()} | "
            f"len(sup)={sup.size}, len(que)={que.size}, N={N}"
        )

    # 2) cover exactly all indices 0..N-1
    un = np.union1d(sup, que)
    if un.size != N:
        missing = sorted(set(range(N)).difference(un.tolist()))
        extra   = sorted(set(un.tolist()).difference(range(N)))
        raise RuntimeError(
            f"[{task_key}] Support∪Query size {un.size} != N={N}. "
            f"Missing={missing[:20]}{'…' if len(missing)>20 else ''} "
            f"Extra={extra[:20]}{'…' if len(extra)>20 else ''}"
        )

    # 3) no duplicates within each side
    if np.unique(sup).size != sup.size:
        dup = [int(i) for i, c in zip(*np.unique(sup, return_counts=True)) if c > 1]
        raise RuntimeError(f"[{task_key}] Duplicate indices in support: {dup}")

    if np.unique(que).size != que.size:
        dup = [int(i) for i, c in zip(*np.unique(que, return_counts=True)) if c > 1]
        raise RuntimeError(f"[{task_key}] Duplicate indices in query: {dup}")

    # 4) optional: sanity that sizes add up
    if sup.size + que.size != N:
        raise RuntimeError(
            f"[{task_key}] len(sup)+len(que)={sup.size+que.size} != N={N} "
            f"(len(sup)={sup.size}, len(que)={que.size})"
        )
    
# factory so we can close over a base_seed
def make_collate_fp(n_support: int, base_seed: int, strategy="quantile", n_bins=3, n_clusters=None):
    def collate(batch):
        import numpy as np
        from torch.utils.data import get_worker_info
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0

        x_sup_list, y_sup_list, x_que_list, y_que_list = [], [], [], []
        for task in batch:
            xs, ys = [], []
            for x, y in task:
                xs.append(x if isinstance(x, torch.Tensor) else torch.tensor(x))
                ys.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
            xs = torch.stack(xs)
            ys = torch.stack([y.view(1) if y.ndim == 0 else y for y in ys]).view(-1)

            # draw a per-task seed deterministically from (base_seed, worker_id)
            rs = int(np.random.default_rng((base_seed ^ (wid + 1)) & 0xFFFFFFFF).integers(0, 2**31-1))
            sup_idx, que_idx = balanced_split(
                xs, ys, n_support=n_support,
                strategy=strategy, n_bins=n_bins, n_clusters=n_clusters,
                random_state=rs,
            )

            x_sup_list.append({"x": xs[sup_idx]}); y_sup_list.append(ys[sup_idx])
            x_que_list.append({"x": xs[que_idx]}); y_que_list.append(ys[que_idx])

        y_sup_tensor = torch.stack(y_sup_list)
        y_que_tensor = torch.stack(y_que_list)
        return x_sup_list, y_sup_tensor, x_que_list, y_que_tensor
    return collate


def make_collate_gnn(n_support: int, base_seed: int, strategy=None, n_bins=3, n_clusters=None):
    def collate(batch):
        import numpy as np
        from torch.utils.data import get_worker_info
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0

        x_sup_list, y_sup_list, x_que_list, y_que_list = [], [], [], []
        for task in batch:
            smiles, xds, ys = [], [], []
            for smi, x_d, y in task:
                smiles.append(smi)
                xds.append(x_d if isinstance(x_d, torch.Tensor) else torch.tensor(x_d))
                ys.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
            X_d = torch.stack(xds)
            Y   = torch.stack([y.view(1) if y.ndim == 0 else y for y in ys]).view(-1)

            rs = int(np.random.default_rng((base_seed ^ (wid + 1)) & 0xFFFFFFFF).integers(0, 2**31-1))
            sup_idx, que_idx = balanced_split(
                X_d, Y, n_support=n_support,
                strategy=strategy, n_bins=n_bins, n_clusters=n_clusters,
                random_state=rs,
            )

            def pick(lst, idx): return [lst[i] for i in idx.tolist()]
            x_sup_list.append({"smiles": pick(smiles, sup_idx), "X_d": X_d[sup_idx]}); y_sup_list.append(Y[sup_idx])
            x_que_list.append({"smiles": pick(smiles, que_idx), "X_d": X_d[que_idx]}); y_que_list.append(Y[que_idx])

        y_sup_tensor = torch.stack(y_sup_list)
        y_que_tensor = torch.stack(y_que_list)
        return x_sup_list, y_sup_tensor, x_que_list, y_que_tensor
    return collate



def meta_split_collate_fn_fp(batch, shots: int = 10, queries_per_task: int | None = None, *, strategy="quantile"):
    x_sup_list, y_sup_list, x_que_list, y_que_list = [], [], [], []
    debug = []  # collect (task_key, N, |sup|, |que|)
    for task in batch:
        xs, ys = [], []
        for x, y in task:
            xs.append(x if isinstance(x, torch.Tensor) else torch.tensor(x))
            ys.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
        xs = torch.stack(xs)  # [N, D]
        ys = torch.stack([y.view(1) if y.ndim == 0 else y for y in ys]).view(-1)
        N = ys.numel()

        sup_idx, que_idx = balanced_split(xs, ys, n_support=shots, strategy=strategy, n_bins=3, n_clusters=None)
        _check_indices(sup_idx, que_idx, N, getattr(task, "task_key", "?"))

        # Optionally force constant query size
        if queries_per_task is not None:
            if len(que_idx) < queries_per_task:
                raise RuntimeError(f"[{getattr(task,'task_key','?')}] only {len(que_idx)} query samples; "
                                   f"need {queries_per_task}. Task N={N}, shots={shots}.")
            pick = torch.randperm(len(que_idx))[:queries_per_task]
            que_idx = que_idx[pick]

        x_sup = {"x": xs[sup_idx]}
        y_sup = ys[sup_idx]
        x_que = {"x": xs[que_idx]}
        y_que = ys[que_idx]

        debug.append((getattr(task, "task_key", "?"), int(N), int(len(sup_idx)), int(len(que_idx))))
        x_sup_list.append(x_sup); y_sup_list.append(y_sup)
        x_que_list.append(x_que); y_que_list.append(y_que)

    # ---- diagnose before stacking ----
    que_lens = [y.numel() for y in y_que_list]
    sup_lens = [y.numel() for y in y_sup_list]
    if len(set(que_lens)) != 1 or len(set(sup_lens)) != 1:
        print("\n[collate:FP] MISMATCHED LENGTHS PER TASK:")
        for (tid, N, ns, nq), sl, ql in zip(debug, sup_lens, que_lens):
            print(f"  task_id={tid} | N={N} | support={ns} (tensor={sl}) | query={nq} (tensor={ql})")
        raise RuntimeError("[collate:FP] Cannot stack y_* tensors due to length mismatch across tasks.")

    try:
        y_sup_tensor = torch.stack(y_sup_list)  # [B, shots]
        y_que_tensor = torch.stack(y_que_list)  # [B, constant]
    except RuntimeError as e:
        print("\n[collate:FP] STACK ERROR — per-task sizes:")
        for (tid, N, ns, nq) in debug:
            print(f"  task_id={tid} | N={N} | support={ns} | query={nq}")
        raise
    return x_sup_list, y_sup_tensor, x_que_list, y_que_tensor

def meta_split_collate_fn_gnn(batch):
    """
    batch: List[ MoleculeFormTask ]  (each task yields items (smiles, X_d, y))
    returns (x_sup_list, y_sup_tensor, x_que_list, y_que_tensor)
    where x_*_list are list[{"smiles": list[str], "X_d": Tensor}]
    """
    x_sup_list, y_sup_list, x_que_list, y_que_list = [], [], [], []
    for task in batch:
        smiles, xds, ys = [], [], []
        for smi, x_d, y in task:
            smiles.append(smi)
            xds.append(x_d if isinstance(x_d, torch.Tensor) else torch.tensor(x_d))
            ys.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
        X_d = torch.stack(xds)                        # [N, d]
        Y   = torch.stack([y.view(1) if y.ndim == 0 else y for y in ys]).view(-1)

        # split on X_d (continuous), not on SMILES strings
        sup_idx, que_idx = balanced_split(X_d, Y, n_support=10, n_bins=3, n_clusters=None,strategy=None)

        def pick(lst, idx):
            idx = idx.tolist()
            return [lst[i] for i in idx]

        x_sup = {"smiles": pick(smiles, sup_idx), "X_d": X_d[sup_idx]}
        y_sup = Y[sup_idx]
        x_que = {"smiles": pick(smiles, que_idx), "X_d": X_d[que_idx]}
        y_que = Y[que_idx]

        x_sup_list.append(x_sup); y_sup_list.append(y_sup)
        x_que_list.append(x_que); y_que_list.append(y_que)

    y_sup_tensor = torch.stack(y_sup_list)
    y_que_tensor = torch.stack(y_que_list)
    return x_sup_list, y_sup_tensor, x_que_list, y_que_tensor

def inference_collater_fp(batch):
    """
    batch: List[ MoleculeFormTask ]
    We assume tasks_per_batch == 1 for inference (recommended).
    Returns (inputs_dict, y_tensor) where inputs_dict["x"] is [N, D]
    """
    assert len(batch) == 1, "Set tasks_per_batch=1 for inference."
    task = batch[0]

    xs, ys = [], []
    for x, y in task:
        xs.append(x if isinstance(x, torch.Tensor) else torch.tensor(x))
        ys.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    X = torch.stack(xs)                           # [N, D]
    Y = torch.stack([y.view(1) if y.ndim == 0 else y for y in ys]).view(-1)  # [N]
    return {"x": X}, Y

def inference_collater_gnn(batch):
    """
    batch: List[ MoleculeFormTask ]
    Returns (inputs_dict, y_tensor) where inputs_dict has:
       - "smiles": list[str] length N
       - "X_d":   Tensor [N, d_extra]
    """
    assert len(batch) == 1, "Set tasks_per_batch=1 for inference."
    task = batch[0]

    smiles, xds, ys = [], [], []
    for smi, x_d, y in task:
        smiles.append(smi)
        xds.append(x_d if isinstance(x_d, torch.Tensor) else torch.tensor(x_d))
        ys.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    X_d = torch.stack(xds)                        # [N, d_extra]
    Y   = torch.stack([y.view(1) if y.ndim == 0 else y for y in ys]).view(-1)  # [N]
    return {"smiles": smiles, "X_d": X_d}, Y

# ---------------------------
# Inference DataLoader wrapper
# ---------------------------


# ----------------------------
# Loader wrapper
# ----------------------------
class LoadTrainData:
    def __init__(self,
                 train_csv: str,
                 val_csv: str,
                 holdout_csv: str = None,
                 bits: int = 2048,
                 radius: int = 4,
                 tasks_per_batch: int = 8,
                 shots: int = 10,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 featurization: str = "FP",
                 ran_seed: int = 42):
        self.bits = bits
        self.radius = radius
        self.tasks_per_batch = tasks_per_batch
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shots = shots
        self.featurization = featurization 
        

        self.train_ds = MolecularTasks(path=train_csv,  assignTasks=False, bits=self.bits, radius=self.radius, featurization=self.featurization)
        self.val_ds   = MolecularTasks(path=val_csv,    assignTasks=False, bits=self.bits, radius=self.radius, featurization=self.featurization)
        self.holdout_ds = None
        if holdout_csv is not None:
            self.holdout_ds = MolecularTasks(path=holdout_csv, assignTasks=False, bits=self.bits, radius=self.radius, featurization=self.featurization)

        self.seed = int(ran_seed)
        self.shots = shots
        self.featurization = featurization

    def _loader(self, dataset, shuffle: bool):
        if self.featurization == "FP":
            collate = make_collate_fp(n_support=self.shots, base_seed=self.seed, strategy="random", n_bins=3)
        else:
            collate = make_collate_gnn(n_support=self.shots, base_seed=self.seed, strategy="random", n_bins=3)

        g = torch.Generator()
        g.manual_seed(self.seed)   # deterministic sampler/shuffle

        return DataLoader(
            dataset,
            batch_size=self.tasks_per_batch,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            worker_init_fn=_seed_worker,
            generator=g,
        )

    def train_loader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_loader(self):
        return self._loader(self.val_ds, shuffle=False)

    def holdout_loader(self):
        if self.holdout_ds is None:
            return None
        return self._loader(self.holdout_ds, shuffle=False)

    def test_loader(self):
        if self.holdout_ds is None:
            return None
        return self.holdout_loader()

class LoadInfData:
    """
    Slim loader for (train, val, inference) episodes.
    Use tasks_per_batch=1 so each batch is exactly one task.
    """
    def __init__(self,
                 train_csv: str,
                 val_csv: str,
                 inf_csv: str,
                 bits: int = 2048,
                 radius: int = 4,
                 tasks_per_batch: int = 1,
                 num_workers: int = 2,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 featurization: str = "FP",
                 ran_seed: int = 42):
        self.bits = bits
        self.radius = radius
        self.tasks_per_batch = tasks_per_batch
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.featurization = featurization
        self.seed = int(ran_seed)

        # retain paths (for saving)
        self.train_path = train_csv
        self.val_path   = val_csv
        self.inf_path   = inf_csv

        # datasets
        self.train_ds = MolecularTasks(path=train_csv, bits=self.bits, radius=self.radius, featurization=self.featurization)
        self.val_ds   = MolecularTasks(path=val_csv,   bits=self.bits, radius=self.radius, featurization=self.featurization)
        self.inf_ds   = MolecularTasks(path=inf_csv,   bits=self.bits, radius=self.radius, featurization=self.featurization)

    def _loader(self, dataset, shuffle: bool):
        if self.featurization == "FP":
            collate = inference_collater_fp
        else:
            collate = inference_collater_gnn

        g = torch.Generator()
        g.manual_seed(self.seed)

        return DataLoader(
            dataset,
            batch_size=self.tasks_per_batch,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            worker_init_fn=_seed_worker,
            generator=g,
        )

    def train_loader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_loader(self):
        return self._loader(self.val_ds, shuffle=False)

    def inf_loader(self):
        return self._loader(self.inf_ds, shuffle=False)