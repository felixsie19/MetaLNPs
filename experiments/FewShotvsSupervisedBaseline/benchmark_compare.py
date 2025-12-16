import os
import math
import time
import random
import json
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# at the very top of experiments/FewShotvsSupervisedBaseline/benchmark_compare.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]           # .../FewShotLNPs
sys.path.insert(0, str(ROOT))                        # enables "from src...."
sys.path.insert(0, str(ROOT / "src" / "utils" / "chemprop"))      # enables "import chemprop" (vendored)


# --- your project imports ---
from src.data.molecules import LoadTrainData
from src.utils.fewshot_trainer import FewShotTrainer
from src.utils.supervised_trainer import SupervisedTrainer
from src.models.few_shot import MLPRegressor, ChempropModel

# ------------------ Fixed config ------------------
TRAIN_CSV = "/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/data/Processed/siRNAho/train_df_task_nosirna_clean.csv"
VAL_CSV   = "/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/data/Processed/siRNAho/meta_val_stop_df_siRNA_clean.csv"
HOLD_CSV  = "/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/data/Processed/siRNAho/holdout_df_task_sirna2_clean.csv"


ROOT_RUNS = "/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/experiments/FewShotvsSupervisedBaseline"
TB_DIR_FSL = os.path.join(ROOT_RUNS, "runs", "fewshot")          # per-mode subfolders added later
TB_DIR_SUP = os.path.join(ROOT_RUNS, "runs", "supervised")       # per-mode subfolders added later
CKPT_DIR_SUP = os.path.join(ROOT_RUNS, "checkpoints_sup")        # per-mode subfolders added later
CKPT_DIR_FS  = os.path.join(ROOT_RUNS, "checkpoints_fewshot")    # per-mode subfolders added later
RESULTS_ALL  = os.path.join(ROOT_RUNS, "benchmark_results_allseeds_bothmodes.csv")

# data params
BITS = 2048
XTRA_DIM_FP = 5                 # formulation features count in FP mode
IN_DIM_FP = BITS + XTRA_DIM_FP
HIDDEN = 256
TASKS_PER_BATCH = 8
SHOTS = 10
NUM_WORKERS = 4
PIN_MEMORY = True

# few-shot configs
FEWSHOT_CONFIGS = [
    #dict(name="Anil", algorithm="anil", adapt_lr=1e-3, meta_lr=3e-4, adapt_steps=3, episode_norm=False, head_only=True),
    dict(name="FoMAML", algorithm="FoMAML", adapt_lr=1e-3, meta_lr=3e-4, adapt_steps=3, episode_norm=False, head_only=False),
    dict(name="MAML",   algorithm="MAML",   adapt_lr=1e-3, meta_lr=3e-4, adapt_steps=3, episode_norm=False, head_only=False),
    dict(name="MetaSGD",algorithm="MetaSGD",adapt_lr=1e-3, meta_lr=3e-4, adapt_steps=1, episode_norm=False, head_only=False),
]

# seeds
SEEDS = [42, 123, 2025, 7, 999, 12, 654, 233, 32, 111]

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------ Builders ------------------
def build_data(mode: str, random_seed) -> LoadTrainData:
    """mode: 'FP' or 'GNN' -> sets collate/outputs accordingly via featurization flag."""
    return LoadTrainData(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        holdout_csv=HOLD_CSV,
        bits=BITS, radius=4,
        tasks_per_batch=TASKS_PER_BATCH,
        shots=SHOTS,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        featurization=mode,     # "FP" or "GNN"
        ran_seed=random_seed
    )

def probe_xtra_dim_gnn(data: LoadTrainData) -> int:
    """Peek one train batch to get X_d width in GNN mode."""
    x_sup, y_sup, x_que, y_que = next(iter(data.train_loader()))
    d_extra = x_sup[0]["X_d"].shape[1]
    return int(d_extra)

def model_factory(mode: str, gnn_xtra_dim: Optional[int] = None):
    """Return a callable that builds a fresh model for the given mode."""
    if mode == "FP":
        return lambda: MLPRegressor(in_dim=IN_DIM_FP, hidden=HIDDEN)
    elif mode == "GNN":
        if gnn_xtra_dim is None:
            raise ValueError("gnn_xtra_dim must be provided for GNN mode.")
        return lambda: ChempropModel(hidden_dim=300, depth=3, dropout=0.0,
                                     use_norm_agg=False, xtra_dim=gnn_xtra_dim, head_hidden=256)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def head_name_for_mode(mode: str) -> str:
    """Head module name for head-only finetuning."""
    return "net.6" if mode == "FP" else "head"   # ChempropModel uses .head, FP MLP uses .net[6]

# ------------------ Runners ------------------
def run_supervised(data, mode: str, make_model):
    m = make_model()
    sup = SupervisedTrainer(
        model=m, data=data, loss="MSE",
        lr=3e-4, patience=100,
        log_dir=os.path.join(TB_DIR_SUP, mode),
        ckpt_dir=os.path.join(CKPT_DIR_SUP, mode),
        mixed_precision=False
    )
    best_val = sup.fit(epochs=20, eval_every=10)

    # eval on holdout with/without K-shot finetune
    hold_noft = sup.evaluate_meta_loader(
        data.holdout_loader(), k_shot_finetune=False, log_prefix=f"{mode}_holdout_noft"
    )
    head_name = head_name_for_mode(mode)
    hold_ft   = sup.evaluate_meta_loader(
        data.holdout_loader(), k_shot_finetune=True, finetune_steps=3, finetune_lr=1e-3,
        head_only=True, head_name=head_name, log_prefix=f"{mode}_holdout_ft"
    )
    return {
        f"{mode}_Supervised_noFT_val_loss": best_val["loss"],
        f"{mode}_Supervised_noFT_holdout_r2": hold_noft["r2"],
        f"{mode}_Supervised_noFT_holdout_rmse": hold_noft["rmse"],
        f"{mode}_Supervised_FT_holdout_r2": hold_ft["r2"],
        f"{mode}_Supervised_FT_holdout_rmse": hold_ft["rmse"],
    }

def run_fewshot(name: str, data, algo_cfg: Dict[str, Any], mode: str, make_model):
    m = make_model()
    head_name = head_name_for_mode(mode)
    trainer = FewShotTrainer(
        model=m, data=data, loss="MSE",
        algorithm=algo_cfg["algorithm"],
        adapt_lr=algo_cfg["adapt_lr"],
        meta_lr=algo_cfg["meta_lr"],
        adapt_steps=algo_cfg["adapt_steps"],
        episode_norm=algo_cfg.get("episode_norm", False),
        head_only=algo_cfg.get("head_only", False),
        head_name=head_name,                              
        patience=algo_cfg.get("patience", 100),
        log_dir=os.path.join(TB_DIR_FSL, mode, name),
        ckpt_dir=os.path.join(CKPT_DIR_FS, mode, name),
        max_grad_norm=1.0,
        mixed_precision=False,
    )
    best_val = trainer.fit(epochs=20, eval_every=10)
    hold     = trainer.evaluate_split("holdout")

    tag = f"{mode}_{name}"
    return {
        f"{tag}_val_loss": best_val["loss"],
        f"{tag}_val_rmse": best_val["rmse"],
        f"{tag}_val_r2":   best_val["r2"],
        f"{tag}_holdout_loss": hold["loss"],
        f"{tag}_holdout_rmse": hold["rmse"],
        f"{tag}_holdout_r2":   hold["r2"],
    }

# ------------------ Main experiment ------------------
def main():
    os.makedirs(os.path.dirname(RESULTS_ALL), exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    for run_idx, seed in enumerate(SEEDS, 1):
        print(f"\n================ SEED {seed} ({run_idx}/{len(SEEDS)}) ================")
        set_seed(seed)

        for mode in ("FP", "GNN"):
            print(f"\n----- Mode: {mode} -----")
            data = build_data(mode,seed)

            # build model factory per mode
            if mode == "FP":
                make_model = model_factory("FP")
            else:
                gnn_xtra_dim = probe_xtra_dim_gnn(data)
                print(f"[GNN] inferred extra feature dim = {gnn_xtra_dim}")
                make_model = model_factory("GNN", gnn_xtra_dim=gnn_xtra_dim)

            row: Dict[str, Any] = {"seed": seed, "mode": mode}

            print(">> Training supervised baselines ...")
            row.update(run_supervised(data, mode, make_model))

            for cfg in FEWSHOT_CONFIGS:
                print(f">> Training {cfg['name']} ...")
                row.update(run_fewshot(cfg["name"], data, cfg, mode, make_model))

            all_rows.append(row)

    # ---- Save per-seed results ----
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_ALL, index=False)
    print(f"Saved per-seed results to {RESULTS_ALL}")

    # ---- Summaries + simple plots per mode ----
    for mode in ("FP", "GNN"):
        sub = df[df["mode"] == mode].drop(columns=["mode", "seed"])
        summary = sub.agg(["mean", "std"]).T
        summ_csv = os.path.join(ROOT_RUNS, f"benchmark_summary_{mode}.csv")
        summary.to_csv(summ_csv)
        print(f"Saved {mode} summary to {summ_csv}")

        # simple holdout R² bars
        labels = [
            (f"{mode}_Supervised_noFT_holdout_r2", "Supervised (no FT)"),
            (f"{mode}_Supervised_FT_holdout_r2",   "Supervised (K-shot FT)"),
            (f"{mode}_FoMAML_holdout_r2",          "FoMAML"),
            (f"{mode}_MAML_holdout_r2",            "MAML"),
            (f"{mode}_MetaSGD_holdout_r2",         "MetaSGD"),
        ]
        # if a key is missing (rare), skip gracefully
        labels = [(k, n) for (k, n) in labels if k in summary.index]
        if labels:
            methods, means = zip(*[(name, summary.loc[key, "mean"]) for key, name in labels])
            errs = [summary.loc[key, "std"] for key, _ in labels]

            plt.figure(figsize=(8, 4.5))
            plt.bar(methods, means, yerr=errs, capsize=4)
            plt.ylabel("Holdout R²")
            plt.title(f"{mode}: Few-shot vs Supervised — mean±std over {len(SEEDS)} seeds")
            plt.xticks(rotation=20, ha="right")
            plt.grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            out_png = os.path.join(ROOT_RUNS, f"benchmark_holdout_r2_avg_{mode}.png")
            plt.savefig(out_png, dpi=300)
            plt.close()
            print(f"Saved {mode} R² plot to {out_png}")

if __name__ == "__main__":
    main()
