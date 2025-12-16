
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
import sys
# 1) parent directory that contains the 'chemprop' folder
sys.path.insert(0, "/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/src/utils/chemprop")

# 2) your project source root (so 'models', 'utils', etc. are top-level)
sys.path.insert(0, "/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/src")
from models.few_shot import ChempropModel,MLPRegressor
from utils.supervised_trainer import SupervisedTrainer
from utils.fewshot_trainer import FewShotTrainer
from data.molecules import LoadTrainData
import torch

# RDKit (generator API)
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# learn2learn for MAML
try:
    import learn2learn as l2l
except Exception as e:
    raise RuntimeError("This module needs `learn2learn` (pip install learn2learn).") from e


# ==============================
# Shared helpers
# ==============================
def _quantile_bins(y: np.ndarray, n_bins: int = 5) -> Optional[np.ndarray]:
    """Return quantile bins 0..B-1 for regression stratification, else None."""
    y = np.asarray(y).ravel()
    try:
        edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) <= 2:
            return None
        return np.digitize(y, edges[1:-1], right=True)
    except Exception:
        return None


def _episode_normalize(y_sup_t: torch.Tensor, y_ref_t: torch.Tensor):
    """z-score normalize (using support stats)."""
    mu = y_sup_t.mean().item()
    std = y_sup_t.std(unbiased=False).item()
    if std < 1e-8:
        std = 1.0
    return (y_sup_t - mu) / std, (y_ref_t - mu) / std, mu, std


def _load_ckp_into_maml_or_base(maml, base, ckp_path: str, device: torch.device, verbose: bool = False):
    """Try loading a checkpoint into MAML; on failure, try into base model."""
    state = torch.load(ckp_path, map_location=device)
    try:
        maml.load_state_dict(state, strict=True)
        if verbose:
            print(f"[ckp-load] Loaded into MAML: {Path(ckp_path).name}")
        return "maml"
    except Exception:
        try:
            base.load_state_dict(state, strict=False)
            if verbose:
                print(f"[ckp-load] Loaded into base model: {Path(ckp_path).name}")
            return "base"
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {ckp_path}: {e}") from e

def _pearsonr_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if y.size != p.size or y.size == 0:
        return np.nan
    y_m = y - y.mean()
    p_m = p - p.mean()
    denom = np.sqrt((y_m**2).sum()) * np.sqrt((p_m**2).sum())
    if denom <= 0:
        return np.nan
    return float((y_m * p_m).sum() / denom)

def _spearmanr_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # robust ranking with tie handling
    y = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    p = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    return _pearsonr_np(y, p)

def _adapt_fixed_steps(
    learner,
    s_sup: List[str],
    y_sup_n: torch.Tensor,
    Xd_sup: Optional[torch.Tensor],
    steps: int,
    adapt_batch_size: int,
    lossfn: nn.Module,
):
    """Run a fixed number of inner-loop steps on support (with minibatches)."""
    n_sup = y_sup_n.size(0)
    for _ in range(steps):
        if adapt_batch_size >= n_sup:
            idx = torch.arange(n_sup)
        else:
            idx = torch.randint(0, n_sup, (adapt_batch_size,))
        idx = idx.tolist()

        smiles_b = [s_sup[k] for k in idx]
        y_b = y_sup_n[idx]
        Xd_b = (Xd_sup[idx] if Xd_sup is not None else None)

        pred_b = learner(smiles=smiles_b, X_d=Xd_b)
        if pred_b.ndim == 1:
            pred_b = pred_b.unsqueeze(1)
        loss = lossfn(pred_b, y_b)
        learner.adapt(loss)

def _adapt_fixed_steps_fp(
    learner,
    X_sup: torch.Tensor,
    y_sup_n: torch.Tensor,
    steps: int,
    adapt_batch_size: int,
    lossfn: nn.Module,
):
    """Inner-loop for FP (dense vector) models."""
    n_sup = y_sup_n.size(0)
    device = y_sup_n.device
    for _ in range(steps):
        if adapt_batch_size >= n_sup:
            idx = torch.arange(n_sup, device=device)
        else:
            idx = torch.randint(0, n_sup, (adapt_batch_size,), device=device)
        pred_b = learner(X_sup[idx])
        if pred_b.ndim == 1:
            pred_b = pred_b.unsqueeze(1)
        loss = lossfn(pred_b, y_sup_n[idx])
        learner.adapt(loss)


def cv_maml_select_checkpoint(
    path_or_df,
    # features/target
    feature_cols: Optional[List[str]] = None,
    target_col: str = "TARGET",

    n_splits: int = 5,
    support_size: int = 15,
    val_size: int = 5,

    adapt_steps: int = 3,
    adapt_batch_size: int = 64,
    adapt_lr: float = 1e-3,
    first_order: bool = True,
    episode_norm: bool = False,
    # model hypers
    hidden_dim: int = 300,
    depth: int = 3,
    dropout: float = 0.0,
    head_hidden: int = 256,

    ckpt_paths: Optional[List[str]] = None,
    ckpt_dir: Optional[str] = None,
    ckpt_glob: str = "*.pt",

    featurization: str = "GNN",         
    n_bits: int = 2048,                 
    radius: int = 3,                   
    use_counts: bool = False,          

    seed: int = 1337,
    device: Optional[torch.device] = None,
    verbose: bool = False,
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- load data ----
    if isinstance(path_or_df, (str, Path)):
        p = str(path_or_df)
        if p.lower().endswith(".csv"):
            df = pd.read_csv(p)
        elif p.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(p)
        else:
            raise ValueError(f"Unsupported file type: {p}")
    elif isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        raise TypeError("path_or_df must be a path to CSV/XLSX or a pandas DataFrame.")

    if "SMILES" not in df.columns:
        raise ValueError("DataFrame must contain a 'SMILES' column.")
    if target_col not in df.columns:
        raise ValueError(f"DataFrame must contain the target column '{target_col}'.")

    # features
    if feature_cols is None:
        exclude = {"SMILES", target_col, "task_id"}
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    featurization = featurization.upper()
    if featurization not in ("GNN", "FP"):
        raise ValueError("featurization must be 'GNN' or 'FP'.")


    xtra_dim = len(feature_cols)
    y_all = df[target_col].to_numpy(dtype=np.float32)
    N = len(df)

    if featurization == "GNN":
        smiles_all = df["SMILES"].astype(str).tolist()
        X_all = df[feature_cols].to_numpy(dtype=np.float32) if xtra_dim > 0 else np.zeros((N, 0), dtype=np.float32)
        X_all_t = torch.from_numpy(X_all).to(device) if xtra_dim > 0 else None
    else:  

        fps = np.stack([smiles_to_fp(s, n_bits=n_bits, radius=radius, use_counts=use_counts)
                        for s in df["SMILES"].astype(str)])
        if xtra_dim > 0:
            extra = df[feature_cols].to_numpy(dtype=np.float32)
            X_all_fp = np.concatenate([fps, extra], axis=1)
        else:
            X_all_fp = fps
        smiles_all = None
        X_all_t = torch.from_numpy(X_all_fp).to(device)
        in_dim_fp = X_all_fp.shape[1]

    # ---- device ----
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- checkpoints ----
    if ckpt_paths is None:
        if ckpt_dir is None:
            raise ValueError("Provide either ckpt_paths or ckpt_dir.")
        ckpt_paths = sorted([str(p) for p in Path(ckpt_dir).glob(ckpt_glob)])
    if len(ckpt_paths) == 0:
        raise ValueError("No checkpoints found to select from.")

    # metrics holders
    #lossfn = nn.MSELoss()
    lossfn = nn.L1Loss()
    #lossfn = nn.HuberLoss()
    y_pred = np.zeros(N, dtype=np.float32)
    per_fold = []

    # outer CV (stratified if possible)
    bins = _quantile_bins(y_all, n_bins=None)
    if bins is not None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.arange(N), bins)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.arange(N))

    from models.few_shot import ChempropModel  # local import to match your project layout

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, 1):
        train_idx = np.array(train_idx)
        test_idx  = np.array(test_idx)

        # choose support + val from training
        rng = np.random.default_rng(seed + fold_idx)
        rng.shuffle(train_idx)
        if support_size + val_size > len(train_idx):
            raise ValueError(
                f"Fold {fold_idx}: support({support_size}) + val({val_size}) > train size ({len(train_idx)})."
            )
        sup_idx = train_idx[:support_size]
        val_idx = train_idx[support_size:support_size + val_size]
        y_sup  = torch.from_numpy(y_all[sup_idx]).view(-1, 1).to(device)
        y_val  = torch.from_numpy(y_all[val_idx]).view(-1, 1).to(device)
        y_test = torch.from_numpy(y_all[test_idx]).view(-1, 1).to(device)

        # slice data
        if featurization == "GNN":
            s_sup  = [smiles_all[i] for i in sup_idx]
            s_val  = [smiles_all[i] for i in val_idx]
            s_test = [smiles_all[i] for i in test_idx]
            Xd_sup  = (X_all_t[sup_idx]  if X_all_t is not None else None)
            Xd_val  = (X_all_t[val_idx]  if X_all_t is not None else None)
            Xd_test = (X_all_t[test_idx] if X_all_t is not None else None)
        else:  # FP
            X_sup  = X_all_t[sup_idx]
            X_val  = X_all_t[val_idx]
            X_test = X_all_t[test_idx]


        # normalization (support stats only)
        if episode_norm:
            y_sup_n, y_val_n, mu, std = _episode_normalize(y_sup, y_val)
        else:
            y_sup_n, y_val_n = y_sup, y_val
            mu, std = 0.0, 1.0

        # ---------- checkpoint selection on validation ----------
        best_ckp, best_val = None, float("inf")
        for ckp_path in ckpt_paths:
            if featurization == "GNN":
                base = ChempropModel(
                    hidden_dim=hidden_dim, depth=depth, dropout=dropout,
                    xtra_dim=xtra_dim, head_hidden=head_hidden
                ).to(device)
            else:
                # Dummy ANN over FP vectors
                base = MLPRegressor(in_dim=in_dim_fp, hidden=hidden_dim).to(device)

            maml = l2l.algorithms.MAML(base, lr=adapt_lr, first_order=first_order)
            _load_ckp_into_maml_or_base(maml, base, ckp_path, device, verbose=False)
            learner = maml.clone(); learner.train()

            if featurization == "GNN":
                _adapt_fixed_steps(learner, s_sup, y_sup_n, Xd_sup, adapt_steps, adapt_batch_size, lossfn)
                with torch.no_grad():
                    pred_val = learner(smiles=s_val, X_d=Xd_val)
            else:
                _adapt_fixed_steps_fp(learner, X_sup, y_sup_n, adapt_steps, adapt_batch_size, lossfn)
                with torch.no_grad():
                    pred_val = learner(X_val)

            if pred_val.ndim == 1: pred_val = pred_val.unsqueeze(1)
            val_loss = lossfn(pred_val, y_val_n).item()
            if val_loss < best_val:
                best_val, best_ckp = val_loss, ckp_path

        if verbose:
            print(f"[select] Fold {fold_idx}/{n_splits}: {Path(best_ckp).name} (val MSE {best_val:.4f})")

        # ----- final test with chosen ckpt -----
        if featurization == "GNN":
            base = ChempropModel(
                hidden_dim=hidden_dim, depth=depth, dropout=dropout,
                xtra_dim=xtra_dim, head_hidden=head_hidden
            ).to(device)
        else:
            base = MLPRegressor(in_dim=in_dim_fp, hidden=hidden_dim).to(device)

        maml = l2l.algorithms.MAML(base, lr=adapt_lr, first_order=first_order)
        _load_ckp_into_maml_or_base(maml, base, best_ckp, device, verbose=False)

        test_learner = maml.clone(); test_learner.train()
        if featurization == "GNN":
            _adapt_fixed_steps(test_learner, s_sup, y_sup_n, Xd_sup, adapt_steps, adapt_batch_size, lossfn)
        else:
            _adapt_fixed_steps_fp(test_learner, X_sup, y_sup_n, adapt_steps, adapt_batch_size, lossfn)

        test_learner.eval()
        with torch.no_grad():
            if featurization == "GNN":
                pred_test_n = test_learner(smiles=s_test, X_d=Xd_test)
            else:
                pred_test_n = test_learner(X_test)
            if pred_test_n.ndim == 1:
                pred_test_n = pred_test_n.unsqueeze(1)
            pred_test = pred_test_n * std + mu if episode_norm else pred_test_n

        preds_np = pred_test.squeeze(1).cpu().numpy().astype(np.float32)
        y_pred[test_idx] = preds_np
        # per-fold test metrics
        y_test_np = y_test.squeeze(1).cpu().numpy()
        fold_pearson  = float(_pearsonr_np(y_test_np, preds_np))
        fold_spearman = float(_spearmanr_np(y_test_np, preds_np))
        fold_rmse = float(np.sqrt(mean_squared_error(y_test_np, preds_np)))
        fold_mae  = float(mean_absolute_error(y_test_np, preds_np))

        per_fold.append({
            "fold": fold_idx,
            "test_idx": test_idx.tolist(),
            "support_idx": sup_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "chosen_ckpt": Path(best_ckp).name,
            "val_mse": best_val,
            "test_pearson":  fold_pearson,
            "test_spearman": fold_spearman,
            "test_rmse": fold_rmse,
            "test_mae":  fold_mae,
        })


    pearson  = float(_pearsonr_np(y_all, y_pred))
    spearman = float(_spearmanr_np(y_all, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_all, y_pred)))
    mae  = float(mean_absolute_error(y_all, y_pred))
    R2 = float(r2_score(y_all, y_pred))
    metrics: Dict[str, object] = {
        "pearson": pearson, "spearman": spearman,
        "rmse": rmse, "mae": mae, "R2":R2, "n": int(N),
        "feature_cols": feature_cols,
        "per_fold": per_fold,
    }

    return metrics, y_all, y_pred, df


# ==============================
# RF baseline
# ==============================
def smiles_to_fp(
    smiles: str,
    n_bits: int = 4096,
    radius: int = 3,
    use_counts: bool = False,
) -> np.ndarray:
    """
    Morgan fingerprint via RDKit's generator.
    - use_counts=False -> bit vector (ExplicitBitVect) -> dense float32 array {0,1}
    - use_counts=True  -> count fingerprint (UIntSparseIntVect) -> dense float32 array with counts
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        
    )


    if use_counts:
        sv = gen.GetCountFingerprint(mol)  # UIntSparseIntVect
        arr = np.zeros((n_bits,), dtype=np.float32)
        for idx, cnt in sv.GetNonzeroElements().items():
            if idx < n_bits:
                arr[idx] = float(cnt)
        return arr
    else:
        bv = gen.GetFingerprint(mol)  # ExplicitBitVect
        arr = np.zeros((n_bits,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr.astype(np.float32)


def cv_random_forest_baseline(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "TARGET",
    n_splits: int = 5,
    support_size: int = 15,
    val_size: int = 5,
    n_bits: int = 2048,
    radius: int = 2,
    use_counts: bool = True,
    train_on_full_train: bool = True,
    seed: int = 1337,
    n_estimators: int = 600,
    min_samples_leaf: int = 1,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """
    Random Forest baseline using Morgan fingerprints (+ optional numeric features).
    """
    if "SMILES" not in df.columns:
        raise ValueError("DataFrame must contain a 'SMILES' column.")
    if target_col not in df.columns:
        raise ValueError(f"DataFrame must contain the target column '{target_col}'.")

    fps = np.stack([smiles_to_fp(s, n_bits=n_bits, radius=radius, use_counts=use_counts)
                    for s in df["SMILES"]])
    if feature_cols:
        extra = df[feature_cols].to_numpy(dtype=np.float32)
        X_all = np.concatenate([fps, extra], axis=1)
    else:
        X_all = fps

    y_all = df[target_col].to_numpy(dtype=np.float32)
    N = len(df)

    # Splitter
    bins = _quantile_bins(y_all, n_bins=None)
    if bins is not None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.arange(N), bins)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.arange(N))

    y_pred = np.zeros(N, dtype=np.float32)
    per_fold: List[Dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, 1):
        train_idx = np.array(train_idx)

        if train_on_full_train:
            train_sub_idx = train_idx
        else:
            rng = np.random.default_rng(seed + fold_idx)
            rng.shuffle(train_idx)
            if support_size + val_size > len(train_idx):
                raise ValueError(f"support+val > train in fold {fold_idx}")
            sup_idx = train_idx[:support_size]
            val_idx = train_idx[support_size:support_size + val_size]
            train_sub_idx = np.concatenate([sup_idx, val_idx])

        X_train, y_train = X_all[train_sub_idx], y_all[train_sub_idx]
        X_test,  y_test  = X_all[test_idx],       y_all[test_idx]

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            bootstrap=True,
            random_state=seed + fold_idx,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            rf.fit(X_train, y_train)

        preds = rf.predict(X_test).astype(np.float32)
        y_pred[test_idx] = preds

        per_fold.append({
            "fold": fold_idx,
            "train_size": int(len(train_sub_idx)),
            "test_size":  int(len(test_idx)),
            "rf_pearson":  float(_pearsonr_np(y_test, preds)),
            "rf_spearman": float(_spearmanr_np(y_test, preds)),
            "rf_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "rf_mae":  float(mean_absolute_error(y_test, preds)),
        })


    pearson  = float(_pearsonr_np(y_all, y_pred))
    spearman = float(_spearmanr_np(y_all, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_all, y_pred)))
    mae  = float(mean_absolute_error(y_all, y_pred))
    R2 = float(r2_score(y_all,y_pred))
    metrics = {
        "pearson": pearson, "spearman": spearman,
        "rmse": rmse, "mae": mae, "R2":R2,"n": int(N),
    }

    return metrics, y_all, y_pred, per_fold


def save_maml_cv_results(
    metrics: Dict[str, object],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    out_dir: str,
    prefix: str = "maml_cv",
) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    with open(outp / f"{prefix}_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "per_fold"}, f, indent=2)

    pd.DataFrame(metrics["per_fold"]).to_csv(outp / f"{prefix}_per_fold.csv", index=False)

    per_sample = pd.DataFrame({
        "index": np.arange(len(y_true)),
        "y_true": y_true.astype(float),
        "y_pred": y_pred.astype(float),
        "residual": (y_pred - y_true).astype(float),
    })
    if "SMILES" in df.columns:
        per_sample["SMILES"] = df["SMILES"].astype(str).values
    per_sample.to_csv(outp / f"{prefix}_per_sample.csv", index=False)


def plot_maml_cv_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_fold: List[Dict[str, float]],
    out_dir: str,
    prefix: str = "maml_cv",
    metric: str = "pearson",   
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    col = "test_pearson" if metric.lower().startswith("pear") else "test_spearman"
    label = "Pearson r" if col == "test_pearson" else "Spearman ρ"

    folds = [d["fold"] for d in per_fold]
    vals  = [d.get(col, np.nan) for d in per_fold]

    plt.figure(figsize=(7, 4))
    plt.bar([str(f) for f in folds], vals)
    plt.xlabel("Fold")
    plt.ylabel(label)
    plt.title(f"MAML — Per-fold {label}")
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / f"{prefix}_{metric}_per_fold.png"), dpi=180)
    plt.close()

    # pooled scatter (unchanged)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(f"MAML — y_true vs y_pred (pooled)")
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / f"{prefix}_scatter.png"), dpi=180)
    plt.close()



def save_rf_cv_results(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_fold: List[Dict[str, float]],
    df: pd.DataFrame,
    out_dir: str,
    prefix: str = "rf_cv",
) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    with open(outp / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(per_fold).to_csv(outp / f"{prefix}_per_fold.csv", index=False)

    out_df = pd.DataFrame({
        "index": np.arange(len(y_true)),
        "y_true": y_true.astype(float),
        "y_pred": y_pred.astype(float),
        "residual": (y_pred - y_true).astype(float),
    })
    if "SMILES" in df.columns:
        out_df["SMILES"] = df["SMILES"].astype(str).values
    out_df.to_csv(outp / f"{prefix}_per_sample.csv", index=False)


def plot_rf_cv_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_fold: List[Dict[str, float]],
    out_dir: str,
    prefix: str = "rf_cv",
    metric: str = "pearson",   # 'pearson' or 'spearman'
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    col = "rf_pearson" if metric.lower().startswith("pear") else "rf_spearman"
    label = "Pearson r" if col == "rf_pearson" else "Spearman ρ"

    folds = [d["fold"] for d in per_fold]
    vals  = [d.get(col, np.nan) for d in per_fold]

    plt.figure(figsize=(7, 4))
    plt.bar([str(f) for f in folds], vals)
    plt.xlabel("Fold")
    plt.ylabel(label)
    plt.title(f"Random Forest — Per-fold {label}")
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / f"{prefix}_{metric}_per_fold.png"), dpi=180)
    plt.close()

    # pooled scatter
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("Random Forest — y_true vs y_pred (pooled)")
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / f"{prefix}_scatter.png"), dpi=180)
    plt.close()



def plot_compare_models(
    y_true: np.ndarray,
    y_pred_a: np.ndarray, label_a: str,
    y_pred_b: np.ndarray, label_b: str,
    out_dir: str,
    prefix: str = "compare_models",
) -> None:
    """Overlay scatter of two methods against the same y_true."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    lo = float(min(y_true.min(), y_pred_a.min(), y_pred_b.min()))
    hi = float(max(y_true.max(), y_pred_a.max(), y_pred_b.max()))
    plt.figure(figsize=(5.6, 5.6))
    plt.scatter(y_true, y_pred_a, s=18, alpha=0.6, label=label_a)
    plt.scatter(y_true, y_pred_b, s=18, alpha=0.6, label=label_b)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("Model comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / f"{prefix}_scatter_overlay.png"), dpi=180)
    plt.close()


def run_maml_cv_and_report(path_or_df, out_dir: str, prefix: str = "maml_cv",
                           plot_metric: str = "pearson", featurization: str = "GNN", **maml_kwargs):
    metrics, y_true, y_pred, df = cv_maml_select_checkpoint(
        path_or_df,
        featurization=featurization,    
        **maml_kwargs
    )
    save_maml_cv_results(metrics, y_true, y_pred, df, out_dir=out_dir, prefix=prefix)
    plot_maml_cv_results(y_true, y_pred, metrics["per_fold"], out_dir=out_dir, prefix=prefix, metric=plot_metric)
    return metrics


def run_rf_cv_and_report(df: pd.DataFrame, out_dir: str, prefix: str = "rf_cv", plot_metric: str = "pearson", **rf_kwargs):
    metrics, y_true, y_pred, per_fold = cv_random_forest_baseline(df=df, **rf_kwargs)
    save_rf_cv_results(metrics, y_true, y_pred, per_fold, df, out_dir=out_dir, prefix=prefix)
    plot_rf_cv_results(y_true, y_pred, per_fold, out_dir=out_dir, prefix=prefix, metric=plot_metric)
    return metrics



def plot_r2_bar_with_error(
    model_to_perfold_csv: dict,
    r2_col_map: dict | None = None,
    title: str = "Holdout R² (mean ± std) across CV folds",
    out_path: str = "outputs/compare/maml_vs_rf_r2_bar.png",
    show_values: bool = True,
):
    """
    model_to_perfold_csv: {"MAML": "outputs/maml/maml_ckp_select_per_fold.csv",
                           "RF":   "outputs/rf/rf_baseline_per_fold.csv"}
    r2_col_map: optional override of the R² column per file,
                e.g. {"MAML":"test_r2", "RF":"rf_r2"}
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    names, means, stds = [], [], []
    rows = []

    for name, csv_path in model_to_perfold_csv.items():
        df = pd.read_csv(csv_path)

        # pick the R² column
        if r2_col_map and name in r2_col_map:
            r2_col = r2_col_map[name]
        else:
            # try common column names
            for cand in ["test_r2", "rf_r2", "r2"]:
                if cand in df.columns:
                    r2_col = cand
                    break
            else:
                raise ValueError(f"No R² column found in {csv_path} (looked for test_r2/rf_r2/r2).")

        r2_vals = df[r2_col].to_numpy(dtype=float)
        names.append(name)
        means.append(np.nanmean(r2_vals))
        stds.append(np.nanstd(r2_vals, ddof=1) if len(r2_vals) > 1 else 0.0)

        for i, r in enumerate(r2_vals, 1):
            rows.append({"model": name, "fold": i, "r2": r})

    # Plot
    plt.figure(figsize=(6, 4))
    x = np.arange(len(names))
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, names)
    plt.ylabel("R²")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if show_values:
        for xi, m, s in zip(x, means, stds):
            plt.text(xi, m + (s if s > 0 else 0) + 0.01, f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    # Also return a tidy DataFrame of per-fold values and a summary
    per_fold_df = pd.DataFrame(rows)
    summary = pd.DataFrame({"model": names, "r2_mean": means, "r2_std": stds})
    return per_fold_df, summary

# ---- Correlation summary + plotting from per-sample CSVs ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def _pearsonr_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if y.size != p.size or y.size == 0:
        return np.nan
    y_m = y - y.mean()
    p_m = p - p.mean()
    denom = np.sqrt((y_m**2).sum()) * np.sqrt((p_m**2).sum())
    if denom <= 0:
        return np.nan
    return float((y_m * p_m).sum() / denom)

def _spearmanr_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # use pandas for robust ranking (handles ties)
    y = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    p = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    return _pearsonr_np(y, p)

def _bootstrap_std(y_true: np.ndarray, y_pred: np.ndarray, metric: str, n_boot: int = 2000, seed: int = 1337) -> float:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    n = y.size
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y[idx]; pt = p[idx]
        if metric == "pearson":
            v = _pearsonr_np(yt, pt)
        elif metric == "spearman":
            v = _spearmanr_np(yt, pt)
        elif metric == "r2":
            v = float(r2_score(yt, pt))
        else:
            raise ValueError("metric must be one of {'pearson','spearman','r2'}")
        if np.isfinite(v):
            vals.append(v)
    return float(np.nanstd(vals, ddof=1)) if vals else np.nan

def summarize_from_per_sample_csv(
    model_to_csv: dict,
    metric: str = "pearson",
    n_boot: int = 2000,
    seed: int = 1337,
):
    """
    model_to_csv: {"MAML": "outputs/maml/maml_cv_per_sample.csv",
                   "RF":   "outputs/rf/rf_cv_per_sample.csv"}
    metric: 'pearson', 'spearman', or 'r2' (global over pooled predictions)
    Returns (summary_df, details_dict).
    """
    rows = []
    details = {}
    for name, path in model_to_csv.items():
        df = pd.read_csv(path)
        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()

        if metric == "pearson":
            val = _pearsonr_np(y_true, y_pred)
        elif metric == "spearman":
            val = _spearmanr_np(y_true, y_pred)
        elif metric == "r2":
            val = float(r2_score(y_true, y_pred))
        else:
            raise ValueError("metric must be one of {'pearson','spearman','r2'}")

        std = _bootstrap_std(y_true, y_pred, metric=metric, n_boot=n_boot, seed=seed)
        rows.append({"model": name, f"{metric}": val, f"{metric}_std": std})
        details[name] = {"value": val, "std": std, "n": int(len(y_true))}

    summary = pd.DataFrame(rows)
    return summary, details

def plot_metric_bar_from_per_sample(
    model_to_csv: dict,
    metric: str = "pearson",
    n_boot: int = 2000,
    seed: int = 1337,
    title: str | None = None,
    out_path: str = "outputs/compare/corr_bar.png",
    show_values: bool = True,
):
    """
    Builds a bar plot with bootstrap std error bars for the selected metric
    computed on pooled predictions across all folds.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary, _ = summarize_from_per_sample_csv(model_to_csv, metric=metric, n_boot=n_boot, seed=seed)

    names = summary["model"].tolist()
    vals  = summary[metric].to_numpy(dtype=float)
    stds  = summary[f"{metric}_std"].to_numpy(dtype=float)

    plt.figure(figsize=(6, 4))
    x = np.arange(len(names))
    plt.bar(x, vals, yerr=stds, capsize=6)
    plt.xticks(x, names)
    ylab = {"pearson": "Pearson r", "spearman": "Spearman ρ", "r2": "R²"}[metric]
    plt.ylabel(ylab)
    plt.title(title or f"{ylab} (pooled across folds; ± bootstrap std)")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if show_values:
        for xi, v, s in zip(x, vals, stds):
            if np.isfinite(v):
                plt.text(xi, v + (s if np.isfinite(s) and s > 0 else 0) + 0.01, f"{v:.3f}±{(s if np.isfinite(s) else 0):.3f}",
                         ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return summary
