# MetaLNPs

Meta-learning and active learning for lipid nanoparticle (LNP) response prediction under **data-scarce** regimes.

This repository contains code, data, and experiments for:
- **Few-shot learning / meta-learning** pipelines
- **Supervised baselines**
- **Active learning** strategies (e.g., RF, RND, uncertainty-driven sampling)
- **Inference** on internal and external lipid libraries

The overall goal is to predict biological responses (e.g., **mRNA delivery efficacy**) across **multiple cell lines** and experimental conditions using limited experimental data.

---

## Table of contents
- [Repository at a glance](#repository-at-a-glance)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Running experiments](#running-experiments)
  - [Few-shot vs supervised benchmark](#few-shot-vs-supervised-benchmark)
  - [Active learning](#active-learning)
  - [Inference on custom lipids](#inference-on-custom-lipids)
- [Data](#data)
- [Outputs and results](#outputs-and-results)
- [Reproducibility](#reproducibility)
- [Project status](#project-status)


---

## Repository at a glance

**Key features**
- Few-shot learning for LNP response prediction
- Meta-learning-based training pipelines
- Supervised learning baselines
- Active learning strategies (RF, RND, uncertainty-based sampling)
- Support for multiple datasets and cell lines
- Reproducible experiment tracking and result aggregation

---

## Installation

### 1) Clone the repository
```bash
git clone https://github.com/felixsie19/MetaLNPs.git
cd MetaLNPs
```

### 2) Create and activate the conda environment
```bash
conda env create -f py310.yml
conda activate metalnps
```

> Tip: If you run into dependency issues, try creating a fresh conda installation and ensure your conda base is up to date.

---

## Quickstart

Run one of the example experiment scripts:

```bash
python experiments/FewShotvsSupervisedBaseline/benchmark_compare.py
```

Outputs (metrics, plots, and intermediate artifacts) are written into the corresponding experiment folder.

---

## Running experiments

### Few-shot vs supervised benchmark
```bash
python experiments/FewShotvsSupervisedBaseline/benchmark_compare.py
```

### Active learning
```bash
python experiments/ActiveLearning/run_AL.py
```

### Inference on custom lipids
```bash
python experiments/OwnLipidsInference/run_lipid_inf.py
```

---

## Data

The `data/` directory contains:
- **Raw datasets** (e.g., Witten dataset)
- **Processed datasets** used for training and evaluation
- **Inference datasets** for internal and external lipid libraries

Some datasets are provided in reduced `*_small` versions for quick testing and debugging.

> Note: If you are missing certain data files (e.g., due to licensing or size constraints), add them to the expected `data/` subfolders following the same naming conventions used by the scripts.

---

## Outputs and results

Results, plots, and intermediate outputs are stored within each experiment directory, typically including:
- Performance metrics (`.csv` / `.xlsx`)
- Model checkpoints
- Visualizations (e.g., UMAPs, performance curves)

---

## Reproducibility

- Experiments are **seeded** where applicable
- Conda environment is provided via `py310.yml`
- A **SLURM script** is included for HPC execution

---

## Project status

**Research prototype / experimental codebase.**  
This repository accompanies ongoing research and is intended for **reproducibility and extension** rather than production use.

