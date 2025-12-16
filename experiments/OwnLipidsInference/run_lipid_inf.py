import pandas as pd
import numpy as np, pandas as pd
from OwnLipids import run_maml_cv_and_report, run_rf_cv_and_report
from OwnLipids import plot_compare_models,plot_r2_bar_with_error,plot_metric_bar_from_per_sample
# put at the very top of run_lipid_inf.py
import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)  # safe even if MPLBACKEND set

runMAML=True
run_RF=True
compare=True



data_path = "/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/data/Inference/Lasse_lipids_MDAsiRNA.xlsx"


df = pd.read_excel(data_path)

if runMAML == True:

    maml_metrics = run_maml_cv_and_report(
        data_path,
        out_dir="outputs/maml",
        prefix="maml_ckp_select",

        feature_cols=["Cationic_Lipid_to_mRNA_weight_ratio","Cationic_Lipid_Mol_Ratio",
                    "Phospholipid_Mol_Ratio","Cholesterol_Mol_Ratio","PEG_Lipid_Mol_Ratio"],
        target_col="TARGET",
        featurization="GNN",
        n_splits=5, support_size=9, val_size=3,
        adapt_steps=3, adapt_batch_size=64, adapt_lr=1e-3,
        first_order=False, episode_norm=False,
        hidden_dim=300, depth=3, dropout=0.0, head_hidden=256,
        ckpt_dir="/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/experiments/FewShotvsSupervisedBaseline/checkpoints_fewshot/GNN/MAML", ckpt_glob="*.pt",
        seed=1337, verbose=True,
    )

if run_RF==True:
    # ----- RF baseline with Morgan(+formulation) -----
    rf_metrics = run_rf_cv_and_report(
        df,
        out_dir="outputs/rf",
        prefix="rf_baseline",
        feature_cols=["Cationic_Lipid_to_mRNA_weight_ratio","Cationic_Lipid_Mol_Ratio",
                    "Phospholipid_Mol_Ratio","Cholesterol_Mol_Ratio","PEG_Lipid_Mol_Ratio"],
        target_col="TARGET",
        n_splits=5, support_size=9, val_size=3,  # used only if train_on_full_train=False
        n_bits=2048, radius=2, use_counts=False,
        train_on_full_train=False,  
        seed=1337, n_estimators=500, min_samples_leaf=1,
    )


if compare == True:
    # Pooled Pearson:
    summary_pearson = plot_metric_bar_from_per_sample(
        {
            "MAML": "outputs/maml/maml_ckp_select_per_sample.csv",
            "RF":   "outputs/rf/rf_baseline_per_sample.csv",
        },
        metric="pearson",
        out_path="outputs/compare/pearson_bar.png",
    )
    print(summary_pearson)

    # Pooled Spearman:
    summary_spearman = plot_metric_bar_from_per_sample(
        {
            "MAML": "outputs/maml/maml_ckp_select_per_sample.csv",
            "RF":   "outputs/rf/rf_baseline_per_sample.csv"
        },
        metric="spearman",
        out_path="outputs/compare/spearman_bar.png",
    )