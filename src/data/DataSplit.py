import json, math, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

##############################################################################
# Parameters you may want to adjust
##############################################################################
RAW_CSV      = "../../data/Raw/witten_dataset.csv"
TRAIN_CSV    = "../../data/Processed/train_df_taskCV.csv"
VAL_CSV      = "../../data/Processed/holdout_df_task.csv"
FOLDS_JSON   = "../../data/Processed/train85_folds.json"

GROUP_SIZE   = 20          # 20-shot tasks
TEST_SIZE    = 0.15        # 85/15 split
N_BINS       = 5           # label stratification bins
N_FOLDS      = 5           # CV folds
RNG_SEED     = 42

##############################################################################
def assign_task(group: pd.DataFrame, group_size: int = GROUP_SIZE):
    label = group.name               # the group key (task-defining tuple)
    g     = group.copy()
    g["task_id"] = [f"{label}_{i // group_size}" for i in range(len(g))]
    return g

def make_splits():
    df0   = pd.read_csv(RAW_CSV)
    file  = df0.copy()

    cols_to_group = ["Delivery_target_dendritic_cell",
                    "Delivery_target_generic_cell",
                    "Delivery_target_liver",
                    "Delivery_target_lung",	
                    "Delivery_target_lung_epithelium",	
                    "Delivery_target_macrophage",	
                    "Delivery_target_muscle",	
                    "Delivery_target_spleen",
                    "Helper_lipid_ID_DOPE",	
                    "Helper_lipid_ID_DOTAP",
                    "Helper_lipid_ID_DSPC",	
                    "Helper_lipid_ID_MDOA",
                    "Helper_lipid_ID_None",	
                    "Route_of_administration_in_vitro",
                    "Route_of_administration_intramuscular",
                    "Route_of_administration_intratracheal",
                    "Route_of_administration_intravenous",
                    "Batch_or_individual_or_barcoded_Barcoded",
                    "Batch_or_individual_or_barcoded_Individual",
                    "Cargo_type_mRNA",
                    "Cargo_type_pDNA",	
                    "Cargo_type_siRNA",	
                    "Model_type_A549",
                    "Model_type_BDMC",
                    "Model_type_BMDM",
                    "Model_type_HBEC_ALI",
                    "Model_type_HEK293T",
                    "Model_type_HeLa",	
                    "Model_type_IGROV1",
                    "Model_type_Mouse",
                    "Model_type_RAW264p7","split_name_for_normalization"]

    # 1. ─────────────────────────── add task_id and filter small tasks
    file["task_id"] = file.groupby(cols_to_group).ngroup()
    file = file.groupby("task_id", group_keys=False)\
               .apply(lambda g: assign_task(g, group_size=GROUP_SIZE))

    # keep only full 20-shot tasks
    tall = file["task_id"].value_counts()
    file = file[file["task_id"].isin(tall[tall >= GROUP_SIZE].index)]

    # 2. ─────────────────────────── preprocessing & column selection
    ratio_cols = ["Cationic_Lipid_to_mRNA_weight_ratio",
                  "Cationic_Lipid_Mol_Ratio", "Phospholipid_Mol_Ratio",
                  "Cholesterol_Mol_Ratio",    "PEG_Lipid_Mol_Ratio"]
    file[ratio_cols] = file[ratio_cols] / 100

    keep = ["smiles", "task_id", "quantified_delivery",
            *ratio_cols, *cols_to_group]
    file = file[keep].rename(columns={"smiles": "SMILES",
                             "quantified_delivery": "TARGET"})

    ##########################################################################
    # 3.  ───  85 / 15 HOLD-OUT (groups = split_name_for_normalization)   ───
    ##########################################################################

    parent   = "split_name_for_normalization"
    print("Available columns:", file.columns.tolist())   # debug
    assert parent in file.columns, f"Column {parent} not found!"
    parents  = file[parent].unique()

    gss = GroupShuffleSplit(test_size=TEST_SIZE,
                            n_splits=1,
                            random_state=RNG_SEED)
    train_p, val_p = next(gss.split(parents, groups=parents))

    is_train  = file[parent].isin(parents[train_p])
    train_df  = file[is_train].reset_index(drop=True)
    val_df    = file[~is_train].reset_index(drop=True)

    print(f"85/15 split – train {len(train_df):,},  hold-out {len(val_df):,}")

    ##########################################################################
    # 4.  ───  STRATIFIED 5-FOLD CV INSIDE THE 85 % TRAIN SET              ───
    ##########################################################################
    # 4-a  bin the regression target for stratification
    q = np.linspace(0, 1, N_BINS + 1)
    train_df["y_bin"] = pd.qcut(train_df["TARGET"],
                                q=q,
                                labels=False,
                                duplicates="drop")

    # 4-b  build folds
    try:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS,
                                    shuffle=True,
                                    random_state=RNG_SEED)
    except AttributeError:
        raise RuntimeError("Needs scikit-learn >= 1.3 for StratifiedGroupKFold "
                           "(or fall back to GroupKFold + re-weighting).")

    folds = []
    X_dummy = np.zeros(len(train_df))            # features not needed

    for tr_idx, val_idx in sgkf.split(
            X_dummy,
            y=train_df["y_bin"],
            groups=train_df[parent]):
        folds.append([tr_idx.tolist(), val_idx.tolist()])

    ##########################################################################
    # 5.  ───  SAVE EVERYTHING TO DISK                                     ───
    ##########################################################################
    Path(TRAIN_CSV).parent.mkdir(parents=True, exist_ok=True)
    train_df.drop(columns=["y_bin"]).to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    json.dump(folds, open(FOLDS_JSON, "w"))

    print(f"Saved train/val CSVs and {N_FOLDS} fold indices → {FOLDS_JSON}")

if __name__ == "__main__":
    make_splits()
