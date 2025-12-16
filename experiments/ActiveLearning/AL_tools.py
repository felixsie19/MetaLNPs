import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone


def load_meta_ckpt(trainer, ckpt_path: str, strict: bool = True):
    sd = torch.load(ckpt_path, map_location=trainer.device)
    # common patterns
    if isinstance(sd, dict) and all(isinstance(k, str) for k in sd.keys()) and \
       all(torch.is_tensor(v) for v in sd.values()):
        state_dict = sd                           # raw state_dict (your trainer saves this)
    elif isinstance(sd, dict) and "state_dict" in sd:
        state_dict = sd["state_dict"]            # some training loops wrap it
    elif isinstance(sd, dict) and "meta_state_dict" in sd:
        state_dict = sd["meta_state_dict"]
    else:
        raise ValueError(f"Unrecognized checkpoint format: keys={list(sd.keys())[:5]}")

    missing, unexpected = trainer.meta.load_state_dict(state_dict, strict=strict)
    print(f"Loaded meta ckpt: missing={missing}, unexpected={unexpected}")
    # quick checksum to confirm weights changed
    checksum = sum(p.detach().float().abs().sum().item() for p in trainer.meta.parameters())
    print(f"[meta] param |w| sum = {checksum:.3e}")
    return missing, unexpected

def load_base_ckpt(trainer, ckpt_path: str, strict: bool = True):
    sd = torch.load(ckpt_path, map_location=trainer.device)
    if isinstance(sd, dict) and all(isinstance(k, str) for k in sd.keys()) and \
       all(torch.is_tensor(v) for v in sd.values()):
        state_dict = sd
    elif isinstance(sd, dict) and "state_dict" in sd:
        state_dict = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        state_dict = sd["model"]
    else:
        raise ValueError(f"Unrecognized checkpoint format: keys={list(sd.keys())[:5]}")
    missing, unexpected = trainer.base_model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded base ckpt: missing={missing}, unexpected={unexpected}")
    checksum = sum(p.detach().float().abs().sum().item() for p in trainer.base_model.parameters())
    print(f"[base] param |w| sum = {checksum:.3e}")
    return missing, unexpected
def get_global_fp_arrays(holdout_ds):
    """
    Concatenate all tasks into a single big X [N,D], Y [N].
    Also return an index map to recover (task_key, local_idx).
    """
    Xs, Ys, index_map = [], [], []
    for ti in range(len(holdout_ds)):
        task = holdout_ds[ti]                 # MoleculeFormTask
        task_key = getattr(task, "task_key", f"task_{ti}")
        xs, ys = [], []
        for x, y in task:                     # FP yields (x, y)
            xs.append(x if isinstance(x, torch.Tensor) else torch.tensor(x))
            ys.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
            index_map.append((task_key, len(ys)-1))
        Xs.append(torch.stack(xs))
        Ys.append(torch.stack([y.view(1) if y.ndim == 0 else y for y in ys]).view(-1))
    X = torch.cat(Xs, dim=0).float()
    Y = torch.cat(Ys, dim=0).float()
    return X, Y, index_map

# ---------- 2) Stratified initial support over Y ----------
def stratified_initial_support(Y: torch.Tensor, n_support: int, n_bins: int = 10, seed: int = 42):
    """
    Choose n_support indices covering Y's quantile bins as evenly as possible.
    """
    rng = np.random.default_rng(seed)
    y = Y.detach().cpu().numpy().astype(float)
    # quantile edges (collapse duplicates)
    qs = np.unique(np.quantile(y, np.linspace(0, 1, n_bins+1)))
    # assign each sample to a bin id
    if len(qs) <= 2:
        # degenerate -> random
        sup = rng.choice(len(y), size=n_support, replace=False)
        return np.sort(sup)
    bins = np.digitize(y, qs[1:-1], right=True) 
    sup = []
    # round-robin sampling from bins
    order = rng.permutation(n_bins)
    per_bin_lists = [np.where(bins == b)[0].tolist() for b in range(n_bins)]
    for bl in per_bin_lists:
        rng.shuffle(bl)
    b = 0
    while len(sup) < n_support and any(per_bin_lists):
        bid = order[b % n_bins]
        if per_bin_lists[bid]:
            sup.append(per_bin_lists[bid].pop())
        b += 1
        # avoid infinite loop if many empty bins
        if b > n_bins * (n_support + 2):
            break
    # fill up randomly if needed
    if len(sup) < n_support:
        remaining = sorted(set(range(len(y))) - set(sup))
        extra = rng.choice(remaining, size=n_support - len(sup), replace=False)
        sup.extend(extra.tolist())
    return np.array(sorted(sup), dtype=int)

# ---------- 3) One adaptation + scoring pass on a pool ----------
def adapt_and_score_pool_global_fp(trainer, X, Y, support_idx, pool_idx, use_true_y_for_query_loss=False):
    """
    Adapts from meta-init on support, predicts on pool.
    If use_true_y_for_query_loss=False, the query loss is computed against zeros (no leakage).
    """
    device = trainer.device
    learner = trainer.meta.clone()
    learner.train()

    x_sup = {"x": X[support_idx].to(device)}
    y_sup = Y[support_idx].to(device)

    x_que = {"x": X[pool_idx].to(device)}
    if use_true_y_for_query_loss:
        y_que = Y[pool_idx].to(device)
    else:
        y_que = torch.zeros_like(Y[pool_idx]).to(device)

    # reuse your inner-loop
    with torch.no_grad():
        pass
    que_pred, _ = trainer._adapt_and_predict_single_task(learner, x_sup, y_sup, x_que, y_que)
    return que_pred.detach().squeeze(1).cpu().numpy()

import numpy as np
import pandas as pd

def random_active_learn_global_fp(
    Y: np.ndarray,                  # [N] targets (only used to log 'true')
    support_idx_init: np.ndarray,   # global indices used as the shared seed set
    steps: int = 100,
    seed: int = 2025,
):
    """
    Random baseline: choose the next point uniformly at random from the pool.
    No model is trained. Returns dict with 'history' DataFrame and final indices.
    """
    rng = np.random.default_rng(seed)
    N = int(Y.shape[0])
    support_idx = np.array(sorted(set(support_idx_init.tolist())), dtype=int)
    pool_idx = np.array(sorted(set(range(N)).difference(support_idx.tolist())), dtype=int)

    history = []
    for t in range(steps):
        if len(pool_idx) == 0:
            break
        # pick a random position in the pool
        pos = int(rng.integers(0, len(pool_idx)))
        gidx = int(pool_idx[pos])
        true = float(Y[gidx])

        # update sets
        support_idx = np.concatenate([support_idx, [gidx]])
        pool_idx = np.delete(pool_idx, pos)

        history.append({
            "step": t,
            "global_index": gidx,
            "pred": float("nan"),     # no predictions for random baseline
            "true": true,             # offline info, used by your summaries
            "support_size": int(len(support_idx)),
            "pool_size": int(len(pool_idx)),
            "pool_rmse": float("nan"),
            "pool_r2": float("nan"),
        })

    return {
        "history": pd.DataFrame(history),
        "final_support_idx": support_idx,
        "remaining_pool_idx": pool_idx,
    }


# ---------- 4) Global AL loop (greedy best-predicted) ----------
def active_learn_global_fp(
    trainer,
    holdout_ds,
    init_support_size: int = 10,
    steps: int = 50,
    objective: str = "max",           # "max" or "min"
    init_bins: int = 10,
    seed: int = 1337,
    use_true_y_for_query_loss: bool = False,  # set True for offline eval; False for “no leakage”
    eval_offline_metrics: bool = True,        # compute RMSE/R2 on pool each step (requires true Y)
    chunk: int | None = None,                 # score pool in chunks to save memory
):
    """
    Returns:
      dict with history DataFrame and final support/pool indices (global, across all tasks).
    """
    X, Y, index_map = get_global_fp_arrays(holdout_ds)
    N = int(Y.shape[0])
    if init_support_size >= N:
        raise ValueError(f"init_support_size {init_support_size} must be < N {N}")

    # initial support (stratified), pool is the rest
    support_idx = space_filling_init_support_blend_fp(
    holdout_ds,
    n_support=10,
    bits=2048,      
    alpha=0.8,      
    seed=1337,
    )
    pool_mask = np.ones(N, dtype=bool); pool_mask[support_idx] = False
    pool_idx = np.arange(N, dtype=int)[pool_mask]

    rng = np.random.default_rng(seed)
    history = []

    for t in range(steps):
        if len(pool_idx) == 0:
            break

        # predict on pool (optionally in chunks)
        if chunk is None:
            preds = adapt_and_score_pool_global_fp(trainer, X, Y, support_idx, pool_idx,
                                                   use_true_y_for_query_loss=use_true_y_for_query_loss)
        else:
            preds_parts = []
            for sub in np.array_split(pool_idx, max(1, int(np.ceil(len(pool_idx)/chunk)))):
                if len(sub) == 0: continue
                p = adapt_and_score_pool_global_fp(trainer, X, Y, support_idx, sub,
                                                   use_true_y_for_query_loss=use_true_y_for_query_loss)
                preds_parts.append(p)
            preds = np.concatenate(preds_parts, axis=0)

        # choose best
        best_pos = int(np.argmax(preds) if objective == "max" else np.argmin(preds))
        chosen_global_idx = int(pool_idx[best_pos])
        chosen_pred = float(preds[best_pos])
        chosen_true = float(Y[chosen_global_idx].item())

        # add to support, remove from pool
        support_idx = np.concatenate([support_idx, [chosen_global_idx]])
        pool_idx = np.delete(pool_idx, best_pos)

        # optional offline metrics on remaining pool
        if eval_offline_metrics and len(pool_idx) > 0:
            # recompute on remaining pool for diagnostics
            eval_preds = adapt_and_score_pool_global_fp(trainer, X, Y, support_idx, pool_idx,
                                                        use_true_y_for_query_loss=True)
            y_true_pool = Y[pool_idx].cpu().numpy()
            rmse = float(np.sqrt(np.mean((eval_preds - y_true_pool) ** 2)))
            ss_res = float(np.sum((eval_preds - y_true_pool) ** 2))
            ss_tot = float(np.sum((y_true_pool - y_true_pool.mean()) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        else:
            rmse, r2 = float("nan"), float("nan")

        tk, li = index_map[chosen_global_idx]
        history.append({
            "step": t,
            "global_index": chosen_global_idx,
            "task_key": tk,
            "task_local_idx": li,
            "pred": chosen_pred,
            "true": chosen_true,            # offline-only info
            "support_size": int(len(support_idx)),
            "pool_size": int(len(pool_idx)),
            "pool_rmse": rmse,
            "pool_r2": r2,
        })

    return {
        "history": pd.DataFrame(history),
        "final_support_idx": support_idx,
        "remaining_pool_idx": pool_idx,
        "index_map": index_map,
    }


def space_filling_init_support_blend_fp(
    holdout_ds,
    n_support: int = 10,
    *,
    bits: int = 2048,        # your Morgan FP size
    alpha: float = 0.8,      # weight for Tanimoto; (1-alpha) for formulation distance
    seed: int = 130,
):
    """
    Space-filling selection using blended distance:
      d = alpha * (1 - tanimoto on FP bits) + (1-alpha) * Euclidean on standardized formulation.
    Returns global indices into concatenated holdout set.
    """
    # ---- build X and split into [FP | FORM] ----
    Xs, idx_map = [], []
    for ti in range(len(holdout_ds)):
        task = holdout_ds[ti]
        for li, (x, y) in enumerate(task):
            v = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
            Xs.append(v)
            idx_map.append((holdout_ds.task_keys[ti], li))
    X = np.stack(Xs).astype(np.float32)             # [N, D]
    N, D = X.shape
    if bits >= D:
        raise ValueError(f"bits={bits} >= D={D}; set bits to your FP length.")
    if n_support >= N:
        raise ValueError(f"n_support ({n_support}) must be < N ({N}).")

    X_fp   = (X[:, :bits] > 0).astype(np.float32)   # binarize to be safe
    X_form = X[:, bits:].astype(np.float32)

    # standardize formulation block
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_form = scaler.fit_transform(X_form)

    # precompute popcounts for fast Tanimoto
    pop = X_fp.sum(axis=1)                          # [N], float

    # helper: Tanimoto distance to a single candidate j for all i
    def tanimoto_dist_to(j):
        dot = (X_fp @ X_fp[j])                      # [N]
        denom = pop + pop[j] - dot
        # avoid divide-by-zero if both are zero-vectors
        sim = np.where(denom > 0, dot / denom, 0.0)
        return 1.0 - sim

    # helper: Euclidean distance to formulation vector j for all i
    def form_dist_to(j):
        diff = Z_form - Z_form[j]
        return np.sqrt(np.sum(diff * diff, axis=1))

    # first point: farthest from formulation+fp centroid approx
    # use Euclidean in concatenated (scaled) space as heuristic for first two seeds
    Z_all = np.concatenate([X_fp, Z_form], axis=1)
    mu = Z_all.mean(axis=0, keepdims=True)
    d0 = np.linalg.norm(Z_all - mu, axis=1)
    first = int(np.argmax(d0))
    selected = [first]

    # second: farthest by blended distance from first
    d_fp  = tanimoto_dist_to(first)
    d_frm = form_dist_to(first)
    dmin  = alpha * d_fp + (1 - alpha) * d_frm
    second = int(np.argmax(dmin))
    selected.append(second)

    # update min blended distance for all i wrt current selected set
    d_fp  = np.minimum(d_fp,  tanimoto_dist_to(second))
    d_frm = np.minimum(d_frm, form_dist_to(second))
    dmin  = alpha * d_fp + (1 - alpha) * d_frm

    # greedy add the farthest points
    for _ in range(n_support - 2):
        nxt = int(np.argmax(dmin))
        selected.append(nxt)

        # update min distances with new center
        d_fp  = np.minimum(d_fp,  tanimoto_dist_to(nxt))
        d_frm = np.minimum(d_frm, form_dist_to(nxt))
        dmin  = alpha * d_fp + (1 - alpha) * d_frm

    return np.array(selected, dtype=int)

def get_global_fp_arrays(holdout_ds):
    Xs, Ys, index_map = [], [], []
    for ti in range(len(holdout_ds)):
        task = holdout_ds[ti]
        task_key = getattr(task, "task_key", f"task_{ti}")
        xs, ys = [], []
        for x, y in task:
            xs.append(x if isinstance(x, torch.Tensor) else torch.tensor(x))
            ys.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
            index_map.append((task_key, len(ys)-1))
        Xs.append(torch.stack(xs))
        Ys.append(torch.stack([y.view(1) if y.ndim == 0 else y for y in ys]).view(-1))
    X = torch.cat(Xs, dim=0).float()
    Y = torch.cat(Ys, dim=0).float()
    return X, Y, index_map

# --- define actives by top percentile or by threshold ---
def active_mask_from_percentile(y: np.ndarray, top_p: float = 0.10) -> np.ndarray:
    thr = np.quantile(y, 1.0 - top_p)
    return y >= thr

def active_mask_from_threshold(y: np.ndarray, thr: float) -> np.ndarray:
    return y >= thr

# --- curves over the acquisition trajectory (selection order) ---
def best_so_far_curve(y_sel: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(y_sel)

def simple_regret_curve(y_sel: np.ndarray, y_star: float) -> np.ndarray:
    return y_star - best_so_far_curve(y_sel)

def hits_curve(selected_idx: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    is_hit = active_mask[selected_idx].astype(int)
    return np.cumsum(is_hit)

# --- Enrichment Factor and NEF at k ---
def ef_at_k(selected_idx: np.ndarray, active_mask: np.ndarray, k: int) -> float:
    k = int(min(k, len(selected_idx)))
    hits_k = int(active_mask[selected_idx[:k]].sum())
    p = float(active_mask.mean()) if active_mask.size > 0 else 0.0
    if p <= 0: 
        return float('nan')
    return hits_k / (p * k)  # random baseline EF = 1.0

def nef_at_k(selected_idx: np.ndarray, active_mask: np.ndarray, k: int) -> float:
    ef = ef_at_k(selected_idx, active_mask, k)
    m = int(active_mask.sum())              # number of actives
    p = float(active_mask.mean())
    if p <= 0 or k <= 0:
        return float('nan')
    ef_max = min(k, m) / (p * k)            # theoretical max EF@k
    if ef_max <= 1.0:                        # avoid div by zero when no enrichment possible
        return float('nan')
    return (ef - 1.0) / (ef_max - 1.0)       # 0 = random, 1 = perfect

# --- AUAC / AULC (area under cumulative hits curve), normalized to [0,1] ---
def auac_normalized(selected_idx: np.ndarray, active_mask: np.ndarray) -> float:
    if active_mask.sum() == 0:
        return float('nan')
    c = hits_curve(selected_idx, active_mask).astype(float)  # length T
    # area under curve via trapezoids, normalized by max possible area
    # max area: put all hits first -> sum_{i=1..min(T,m)} i
    T = len(selected_idx)
    m = int(active_mask.sum())
    auc = np.trapz(c, dx=1.0)
    max_auc = (min(T, m) * (min(T, m) + 1)) / 2.0 + max(0, T - m) * m
    return float(auc / max_auc)

# --- nDCG@k using true y as gains (shifted non-negative) ---
def ndcg_at_k(selected_idx: np.ndarray, y: np.ndarray, k: int) -> float:
    k = int(min(k, len(selected_idx)))
    gains = y[selected_idx[:k]].astype(float)
    # shift to non-negative
    gains = gains - gains.min()
    # DCG
    denom = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains / denom))
    # IDCG from ideal top-k
    topk = np.sort(y)[-k:][::-1] - y.min()
    idcg = float(np.sum(topk / denom))
    return dcg / idcg if idcg > 0 else float('nan')

# --- Summary over an AL history ---
def summarize_al_metrics(history_df: pd.DataFrame, Y: np.ndarray,
                         active_def: str = "percentile", top_p: float = 0.10,
                         ks: tuple = (5, 10, 20, 50, 100)) -> dict:
    sel = history_df["global_index"].to_numpy(dtype=int)
    y_all = np.asarray(Y, dtype=float)
    y_sel = y_all[sel]
    y_star = float(np.max(y_all))
    if active_def == "percentile":
        act = active_mask_from_percentile(y_all, top_p=top_p)
    elif active_def.startswith("thr:"):
        thr = float(active_def.split(":",1)[1])
        act = active_mask_from_threshold(y_all, thr)
    else:
        raise ValueError("active_def must be 'percentile' or 'thr:<value>'")

    # curves
    best_curve = best_so_far_curve(y_sel)
    regret_curve = simple_regret_curve(y_sel, y_star)
    hits = hits_curve(sel, act)
    auac = auac_normalized(sel, act)

    # point metrics at specific k
    rows = []
    for k in ks:
        rows.append({
            "k": k,
            "Hit@k": int(hits[min(k, len(hits)) - 1]) if len(hits) >= 1 else 0,
            "EF@k": ef_at_k(sel, act, k),
            "NEF@k": nef_at_k(sel, act, k),
            "nDCG@k": ndcg_at_k(sel, y_all, k),
            "best_so_far@k": float(best_curve[min(k, len(best_curve)) - 1]) if len(best_curve) else float('nan'),
            "simple_regret@k": float(regret_curve[min(k, len(regret_curve)) - 1]) if len(regret_curve) else float('nan'),
            "yield@k": float(np.mean(y_sel[:k])) if k <= len(y_sel) and k > 0 else float('nan'),
        })
    table = pd.DataFrame(rows)

    return {
        "active_frac": float(act.mean()),
        "auac_normalized": auac,
        "best_curve": best_curve,        # arrays for plotting
        "regret_curve": regret_curve,
        "hits_curve": hits,
        "table": table,                  # tidy summary at chosen k
    }
def summary_to_df(summary: Dict[str, Any],
                  handle_mismatch: str = "drop",
                  limit_rows: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Flatten a summary dict into a per-iteration DataFrame.
    - Scalars and 1D arrays become columns.
    - 2D/ND arrays are DROPPED by default (returned in `extras`) to avoid padding/tiling.
    - If limit_rows is set, truncate to that many rows (no filling beyond K).
    Returns (main_df, extras) where `extras` contains non-conforming entries.
    """
    assert handle_mismatch in {"drop"}, "Only 'drop' (no filling) is allowed per your request."

    # Decide nrows from the longest 1D or the first-dim of any 2D entry; if none, set to 1.
    nrows = None
    for v in summary.values():
        a = np.asarray(v)
        if a.ndim == 1:
            nrows = len(a) if nrows is None else max(nrows, len(a))
        elif a.ndim >= 2:
            nrows = a.shape[0] if nrows is None else max(nrows, a.shape[0])
    if nrows is None:
        nrows = 1

    # Apply optional cap
    if limit_rows is not None:
        nrows = min(nrows, int(limit_rows))

    flat = {}
    extras = {}

    for k, v in summary.items():
        a = np.asarray(v)

        if a.ndim == 0:  # scalar → broadcast to nrows
            flat[k] = np.repeat(a.item(), nrows)

        elif a.ndim == 1:
            if len(a) >= nrows:
                flat[k] = a[:nrows]
            else:
                # shorter vectors are dropped (no filling beyond available length)
                extras[k] = a

        else:
            # 2D or ND → drop to extras (you don't want to fill beyond K)
            extras[k] = a

    df = pd.DataFrame(flat)
    return df, extras


def make_kmetrics_df(obj: Any,
                     source_key: str = "table",
                     columns=("k","Hit@k","EF@k","NEF@k","nDCG@k",
                              "best_so_far@k","simple_regret@k","yield@k")) -> pd.DataFrame:
    """
    Build the k-metrics DataFrame from either:
      - a dict containing obj[source_key] as a 2D array-like, or
      - a DataFrame passed directly as `obj`.
    Column headers are set exactly as requested.
    """
    if isinstance(obj, pd.DataFrame):
        kdf = obj.copy()
        # If incoming headers differ, realign them if shapes match
        if len(kdf.columns) != len(columns):
            kdf.columns = list(columns)[:len(kdf.columns)]
        else:
            kdf.columns = list(columns)
        return kdf

    if isinstance(obj, dict) and source_key in obj:
        arr = np.asarray(obj[source_key])
        if arr.ndim != 2:
            raise ValueError(f"'{source_key}' must be 2D, got shape {arr.shape}")
        kdf = pd.DataFrame(arr, columns=list(columns)[:arr.shape[1]])
        return kdf

    raise ValueError(f"Could not construct k-metrics DataFrame (missing key '{source_key}' or wrong type).")


def write_results_excel(path: str,
                        runs_df: pd.DataFrame,
                        kmetrics_df: pd.DataFrame,
                        runs_sheet: str = "runs",
                        km_sheet: str = "k_metrics") -> None:
    """
    Write both DataFrames to a single Excel file with minimal, clean formatting.
    - No CSV is produced.
    - No padding is done; only exact data is written.
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Sheet 1: runs
        runs_df.to_excel(writer, index=False, sheet_name=runs_sheet)
        ws = writer.sheets[runs_sheet]
        ws.freeze_panes = "A2"

        # Sheet 2: k_metrics
        kmetrics_df.to_excel(writer, index=False, sheet_name=km_sheet)
        ws2 = writer.sheets[km_sheet]
        ws2.freeze_panes = "A2"

    print(f"✅ Wrote Excel with sheets '{runs_sheet}' and '{km_sheet}' → {path}")

def rf_active_learn_global_fp(
    X: np.ndarray,                  # [N, D] FP features (numpy)
    Y: np.ndarray,                  # [N] targets (numpy)
    support_idx_init: np.ndarray,   # initial support global indices (same used for MAML)
    steps: int = 100,
    objective: str = "max",         # "max" or "min"
    seed: int = 2025,
    retrain_each_step: bool = True, 
    rf: RandomForestRegressor | None = None,
    eval_offline_metrics: bool = True,
):
    """
    Greedy AL with RF: at each step fit RF on current support and add the pool point with best predicted value.
    Returns dict with 'history' DataFrame and final indices.
    """
    rng = np.random.default_rng(seed)
    N = int(Y.shape[0])
    support_idx = np.array(sorted(set(support_idx_init.tolist())), dtype=int)
    pool_idx = np.array(sorted(set(range(N)).difference(support_idx.tolist())), dtype=int)

    if rf is None:
        rf = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)

    # one-shot model (if requested)
    model = clone(rf)
    if not retrain_each_step:
        model.fit(X[support_idx], Y[support_idx])

    history = []
    for t in range(steps):
        if len(pool_idx) == 0:
            break

        if retrain_each_step:
            model = clone(rf)
            model.fit(X[support_idx], Y[support_idx])

        preds = model.predict(X[pool_idx])
        best_pos = int(np.argmax(preds) if objective == "max" else np.argmin(preds))
        best_global = int(pool_idx[best_pos])
        best_pred = float(preds[best_pos])
        best_true = float(Y[best_global])

        # update sets
        support_idx = np.concatenate([support_idx, [best_global]])
        pool_idx = np.delete(pool_idx, best_pos)

        # optional offline diagnostics on the remaining pool
        if eval_offline_metrics and len(pool_idx) > 0:
            pool_preds = model.predict(X[pool_idx])
            y_true_pool = Y[pool_idx]
            rmse = float(np.sqrt(np.mean((pool_preds - y_true_pool) ** 2)))
            ss_res = float(np.sum((pool_preds - y_true_pool) ** 2))
            ss_tot = float(np.sum((y_true_pool - y_true_pool.mean()) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        else:
            rmse, r2 = float("nan"), float("nan")

        history.append({
            "step": t,
            "global_index": best_global,
            "pred": best_pred,
            "true": best_true,
            "support_size": int(len(support_idx)),
            "pool_size": int(len(pool_idx)),
            "pool_rmse": rmse,
            "pool_r2": r2,
        })

    return {
        "history": pd.DataFrame(history),
        "final_support_idx": support_idx,
        "remaining_pool_idx": pool_idx,
    }
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _load_kmetrics(excel_path: str, sheet_name: str = "k_metrics") -> pd.DataFrame:
    """
    Load k-metrics from an Excel file and return a tidy DataFrame with columns ['k','hits'].
    Tries to be tolerant to small header differences.
    """
    xls = pd.ExcelFile(excel_path)
    # pick the k_metrics sheet (case-insensitive), fallback to first sheet
    sheet = None
    for s in xls.sheet_names:
        if s.lower() == sheet_name.lower():
            sheet = s
            break
    if sheet is None:
        sheet = xls.sheet_names[0]

    df = pd.read_excel(excel_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    # find k column
    k_col = None
    for cand in ["k", "K", "n", "step", "iteration", "iters"]:
        for c in df.columns:
            if c.lower() == cand.lower():
                k_col = c
                break
        if k_col:
            break
    if k_col is None:
        for c in df.columns:
            if "k" in c.lower():
                k_col = c
                break

    # find hits column (Hit@k etc.)
    hit_col = None
    for cand in ["Hit@k", "hits@k", "hit_at_k", "hits", "Hit"]:
        for c in df.columns:
            if c.lower().replace(" ", "") == cand.lower().replace(" ", ""):
                hit_col = c
                break
        if hit_col:
            break
    if hit_col is None:
        for c in df.columns:
            if "hit" in c.lower():
                hit_col = c
                break

    if k_col is None or hit_col is None:
        raise ValueError(
            f"Could not locate 'k' and 'Hit@k' columns in {excel_path}. "
            f"Columns found: {list(df.columns)}"
        )

    km = df[[k_col, hit_col]].copy()
    km.columns = ["k", "hits"]
    km["k"] = pd.to_numeric(km["k"], errors="coerce")
    km["hits"] = pd.to_numeric(km["hits"], errors="coerce")
    km = km.dropna().sort_values("k")
    return km

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ── helpers reused ─────────────────────────────────────────────
def _extract_fp_form(holdout_ds, bits: int = 2048):
    Xs, Ys, task_keys = [], [], []
    for ti in range(len(holdout_ds)):
        task = holdout_ds[ti]
        tkey = getattr(holdout_ds, "task_keys", None)
        for (x, y) in task:
            v = x.numpy() if hasattr(x, "numpy") else np.asarray(x, dtype=np.float32)
            Xs.append(v.astype(np.float32))
            if np.ndim(y) == 0:
                Ys.append(float(y))
            else:
                y_arr = np.asarray(y).ravel()
                Ys.append(float(y_arr[0]) if y_arr.size else np.nan)
            if tkey is None:
                task_keys.append(f"task_{ti}")
            else:
                # tkey may be list[str] or similar
                task_keys.append(tkey[ti] if isinstance(tkey, (list, tuple)) else str(tkey))
    X = np.stack(Xs).astype(np.float32)
    assert bits <= X.shape[1], f"bits={bits} > D={X.shape[1]}"
    X_fp   = (X[:, :bits] > 0).astype(np.float32)
    X_form = X[:, bits:].astype(np.float32)
    return X_fp, X_form, np.array(Ys, dtype=float), np.array(task_keys, dtype=object)

def _tanimoto_dist_matrix(Xb: np.ndarray) -> np.ndarray:
    K = Xb @ Xb.T
    pop = Xb.sum(axis=1, keepdims=True)
    denom = pop + pop.T - K
    sim = np.where(denom > 0, K / denom, 0.0)
    return 1.0 - sim

def _euclidean_dist_matrix(Z: np.ndarray) -> np.ndarray:
    G = Z @ Z.T
    nrm = np.sum(Z*Z, axis=1, keepdims=True)
    D2 = np.maximum(nrm + nrm.T - 2*G, 0.0)
    return np.sqrt(D2, dtype=np.float32)

# ── main: 2D/3D embedding plotter ──────────────────────────────
def plot_initial_support_embedding(
    holdout_ds,
    selected_idx: np.ndarray,
    *,
    bits: int = 2048,
    alpha: float = 0.5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    subsample: int | None = None,       # e.g. 1000 to cap N for speed
    annotate: bool = False,
    color_by: str | None = None,        # None | 'y' | 'task'
    n_components: int = 3,              # <<< set 3 for 3D, 2 for 2D
    title: str | None = None,
    save_path: str | None = None,
):
    """
    Embed the pool with the same blended geometry used for seeding and plot, highlighting seeds.
    Uses UMAP (metric='precomputed') when available; otherwise PCA on concatenated features.
    """
    assert n_components in (2, 3), "n_components must be 2 or 3"

    X_fp, X_form, Y, T = _extract_fp_form(holdout_ds, bits=bits)
    N = X_fp.shape[0]
    idx_all = np.arange(N)

    # Subsample (always keep seeds)
    if subsample is not None and subsample < N:
        rng = np.random.default_rng(random_state)
        keep = set(rng.choice(idx_all, size=subsample, replace=False).tolist())
        keep |= set(map(int, selected_idx))
        keep = np.array(sorted(keep), dtype=int)
        # remap arrays and seed indices
        inv = {g: i for i, g in enumerate(keep.tolist())}
        selected_idx = np.array([inv[int(g)] for g in selected_idx if int(g) in inv], dtype=int)
        X_fp, X_form, Y, T = X_fp[keep], X_form[keep], Y[keep], T[keep]
        idx_all = np.arange(len(keep))

    # Build blended distance
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_form = scaler.fit_transform(X_form)
    D_fp   = _tanimoto_dist_matrix(X_fp)
    D_form = _euclidean_dist_matrix(Z_form)
    D_blend = alpha * D_fp + (1.0 - alpha) * D_form

    # UMAP (precomputed), fallback to PCA
    XY = None
    used = "umap"
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="precomputed",
            random_state=random_state,
            n_components=n_components,
        )
        XY = reducer.fit_transform(D_blend)  # [N, n_components]
    except Exception as e:
        from sklearn.decomposition import PCA
        XY = PCA(n_components=n_components, random_state=random_state).fit_transform(
            np.concatenate([X_fp, Z_form], axis=1)
        )
        used = f"PCA (fallback: {e})"

    # Color encoding
    c = None
    cb_label = None
    if color_by == "y":
        c = Y
        cb_label = "Target (Y)"
    elif color_by == "task":
        _, inv = np.unique(T, return_inverse=True)
        c = inv
        cb_label = "Task ID"

    # Plot (2D or 3D)
    if n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # background pool: a bit smaller points, higher alpha
        sc = ax.scatter(
            XY[:,0], XY[:,1], XY[:,2],
            s=6, alpha=0.5, c=c,
        )

        # seeds: slightly bigger, solid facecolor, black edge
        ax.scatter(
            XY[selected_idx,0], XY[selected_idx,1], XY[selected_idx,2],
            s=70, marker="o", edgecolor="k", linewidths=0.8,
            facecolor="white", alpha=1.0,
            label=f"Initial support (n={len(selected_idx)})",
        )

        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")

    else:
        fig, ax = plt.subplots()

        sc = ax.scatter(
            XY[:,0], XY[:,1],
            s=10, alpha=0.6, c=c,
        )

        ax.scatter(
            XY[selected_idx,0], XY[selected_idx,1],
            s=70, marker="o", edgecolor="k", linewidths=0.8,
            facecolor="white", alpha=1.0,
            label=f"Initial support (n={len(selected_idx)})",
        )

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")


    # Legend / colorbar / title
    if title is None:
        title = f"Initial support embedding ({used}, blended metric)"
    if n_components == 3:
        ax.legend(loc="best", frameon=True)
    else:
        ax.legend(loc="best", frameon=True)
    if c is not None:
        cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.8) if n_components == 2 else plt.colorbar(sc, pad=0.02, shrink=0.8)
        if cb_label: cb.set_label(cb_label)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {"xy": XY, "Y": Y, "tasks": T, "selected_idx": selected_idx, "method": used}

def plot_hits_vs_runs(
    al_xlsx: str,
    rf_xlsx: str,
    *,
    rnd_xlsx: str | None = None,    # ← NEW: optional random baseline file
    align: str = "none",            # 'none' | 'intersection' | 'union'
    label_al: str = "Active Learning",
    label_rf: str = "Random Forest",
    label_rnd: str = "Random",
    save_path: str | None = None,
    show: bool = True,
    ax=None,
):
    al = _load_kmetrics(al_xlsx)
    rf = _load_kmetrics(rf_xlsx)
    rnd = _load_kmetrics(rnd_xlsx) if rnd_xlsx is not None else None

    if align == "intersection":
        ks = al["k"].values
        ks = np.intersect1d(ks, rf["k"].values)
        if rnd is not None:
            ks = np.intersect1d(ks, rnd["k"].values)
        al_plot = al[al["k"].isin(ks)]
        rf_plot = rf[rf["k"].isin(ks)]
        rnd_plot = rnd[rnd["k"].isin(ks)] if rnd is not None else None
    else:
        al_plot, rf_plot, rnd_plot = al, rf, rnd

    created_fig = False
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        created_fig = True

    ax.plot(al_plot["k"], al_plot["hits"], marker="o", label=label_al)
    ax.plot(rf_plot["k"], rf_plot["hits"], marker="s", label=label_rf)
    if rnd_plot is not None:
        ax.plot(rnd_plot["k"], rnd_plot["hits"], marker="^", linestyle="--", label=label_rnd)

    ax.set_xlabel("Number of runs (k)")
    ax.set_ylabel("Hits")
    ax.set_title("Hits vs Number of Runs")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    if save_path:
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show and created_fig:
        import matplotlib.pyplot as plt
        plt.show()

    return {"al": al, "rf": rf, "rnd": rnd}
