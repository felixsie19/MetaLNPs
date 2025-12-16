import sys
sys.path.insert(0, "/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/src/utils/chemprop")
sys.path.insert(0, "/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/src")
from models.few_shot import ChempropModel,MLPRegressor
from utils.supervised_trainer import SupervisedTrainer
from utils.fewshot_trainer import FewShotTrainer
from data.molecules import LoadTrainData
from AL_tools import random_active_learn_global_fp,load_meta_ckpt, plot_initial_support_embedding, active_learn_global_fp, get_global_fp_arrays,summarize_al_metrics,plot_hits_vs_runs, summary_to_df, make_kmetrics_df, write_results_excel,rf_active_learn_global_fp, space_filling_init_support_blend_fp
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

base_model = MLPRegressor(in_dim=2053)
#base_model = ChempropModel(xtra_dim = 5)

meta_ckpt = "/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/experiments/FewShotvsSupervisedBaseline/checkpoints_fewshot/FP/MAML/best_episode910.pt"


#meta_ckpt= "/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/experiments/FewShotvsSupervisedBaseline/checkpoints_fewshot/GNN/MAML/best_episode1070.pt"
fit = True

data = LoadTrainData(
    train_csv="/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/data/Processed/siRNAho/train_df_task_nosirna_clean.csv",
    val_csv="/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/data/Processed/siRNAho/meta_val_stop_df_siRNA_clean.csv",
    holdout_csv="/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/experiments/ActiveLearning/siRNA_fulltaks.xlsx",
    featurization="FP",              
    tasks_per_batch=8,
    shots=10,                      
    pin_memory=True,
    drop_last=True,
    ran_seed=420,
)

trainer = FewShotTrainer(
    model=base_model,
    data=data,
    algorithm="MAML",
    adapt_lr=1e-3,
    meta_lr=3e-4,
    adapt_steps=3,
    episode_norm=False,
    patience=500,
    log_dir="./runs/adaptcurve_train",
    ckpt_dir="/home/akm/Felix_ML/Lasse_lipids/FewShotLNPs/experiments/FewShotvsSupervisedBaseline/checkpoints_fewshot/FP/MAML",
    mixed_precision=False,
)


if fit == True:
    trainer.fit(epochs=20, eval_every=10, save_tag="best")
else:
    load_meta_ckpt(trainer, meta_ckpt, strict=True)


al_out = active_learn_global_fp(
    trainer=trainer,
    holdout_ds=data.holdout_ds,  
    init_support_size=10,
    steps=50,                   
    objective="max",         
    seed=130,
    use_true_y_for_query_loss=False,   
    eval_offline_metrics=True,         
    chunk=None,                       
    )
# idx and feature/labels for rf baseline
#TODO: Integrate into function
X_t, Y_t, _ = get_global_fp_arrays(data.holdout_ds)
X = X_t.cpu().numpy() if hasattr(X_t, "cpu") else X_t
Y = Y_t.cpu().numpy() if hasattr(Y_t, "cpu") else Y_t


support_idx = space_filling_init_support_blend_fp(
    data.holdout_ds, n_support=10, bits=2048, alpha=0.8, seed=130
)

plot_initial_support_embedding(
    holdout_ds=data.holdout_ds,
    selected_idx=support_idx,
    bits=2048,
    alpha=0.8,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    subsample=None,     
    color_by="None",   
    n_components=3,    
    title="Initial space-filling seeds (UMAP 3D, blended metric)",
    save_path="initial_support_umap_3d.png",
)


rf_out= rf_active_learn_global_fp(
    X=X, Y=Y,
    support_idx_init=support_idx,
    steps=50,
    objective="max",
    seed=2025,
    retrain_each_step=True,  
    rf=RandomForestRegressor(n_estimators=500, random_state=12, n_jobs=-1),
)

rf_out["history"].to_csv("rf_global_fp_history_small.csv", index=False)
al_out["history"].to_csv("al_global_fp_history_small.csv", index=False)

summary = summarize_al_metrics(
    history_df=al_out["history"],
    Y=Y_t.cpu().numpy(),
    active_def="percentile",  
    top_p=0.05,               
    ks=(5, 10, 20, 50)
)
summary_rf   = summarize_al_metrics(rf_out["history"],   Y, active_def="percentile", top_p=0.05, ks=(5, 10, 20, 50))

runs_df, extras = summary_to_df(summary, handle_mismatch="drop", limit_rows=None)
basline_runs_df,extra_baseline=summary_to_df(summary_rf, handle_mismatch="drop", limit_rows=None)

kdf = make_kmetrics_df(summary, source_key="table")
kdf_rf = make_kmetrics_df(summary_rf, source_key="table")

out_xlsx = "AL_results_small.xlsx"
out_rf = "RF_baseline_results_small.xlsx"
write_results_excel(out_xlsx, runs_df, kdf)
write_results_excel(out_rf, basline_runs_df, kdf_rf)

# --- RANDOM BASELINE (same initial support as AL/RF) ---
rnd_out = random_active_learn_global_fp(
    Y=Y,
    support_idx_init=support_idx,   # reuse identical seed set
    steps=150,
    seed=2025,
)

# Save histories (optional, keeps parity with your workflow)
rnd_out["history"].to_csv("rnd_global_fp_history_small.csv", index=False)

# Summaries & Excel (so the plotter can load them)
summary_rnd = summarize_al_metrics(
    history_df=rnd_out["history"],
    Y=Y, active_def="percentile", top_p=0.05, ks=(5, 10, 20, 50)
)
rnd_runs_df, rnd_extras = summary_to_df(summary_rnd, handle_mismatch="drop", limit_rows=None)
kdf_rnd = make_kmetrics_df(summary_rnd, source_key="table")
out_rnd = "RND_baseline_results_small.xlsx"
write_results_excel(out_rnd, rnd_runs_df, kdf_rnd)

# --- Plot all three on one chart ---
plot_hits_vs_runs(
    al_xlsx="AL_results_small.xlsx",
    rf_xlsx="RF_baseline_results_small.xlsx",
    rnd_xlsx="RND_baseline_results_small.xlsx",   # ← NEW
    align="intersection",
    save_path="hits_vs_runs_small.png",
)

