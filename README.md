##MetaLNPs

Meta-learning and active learning for lipid nanoparticle (LNP) response prediction using limited experimental data.

This repository contains code, data, and experiments for few-shot learning, supervised baselines, and active learning applied to lipid nanoparticle (LNP) datasets. The project focuses on predicting biological responses (e.g. mRNA delivery efficacy) across different cell lines and experimental conditions under data-scarce regimes.

##Repository Structure
.
├── data/                     # Raw, processed, and inference datasets
│   ├── Raw/                  # Original datasets
│   ├── Processed/            # Preprocessed data splits
│   └── Inference/            # Datasets used for inference and evaluation
│
├── experiments/              # Experimental pipelines and results
│   ├── ActiveLearning/       # Active learning experiments and baselines
│   ├── FewShotvsSupervisedBaseline/
│   │   └── Benchmarking few-shot vs supervised learning
│   └── OwnLipidsInference/   # Inference on custom lipid datasets
│
├── src/                      # Core source code
│   ├── data/                 # Dataset handling and task construction
│   ├── models/               # Few-shot and supervised models
│   └── utils/                # Training loops and utilities
│
├── py310.yml                 # Conda environment specification
├── fewshot.slurm             # SLURM script for cluster execution
└── README.md

##Key Features

Few-shot learning for LNP response prediction

Meta-learning-based training pipelines

Supervised learning baselines

Active learning strategies (e.g. RF, RND, uncertainty-driven sampling)

Support for multiple datasets and cell lines

Reproducible experiment tracking and result aggregation

##Data

The data/ directory contains:

Raw datasets (e.g. Witten dataset)

Processed datasets used for training and evaluation

Inference datasets for both internal and external lipid libraries

Some datasets are provided in reduced (“_small”) versions for quick testing and debugging.

Installation
##1. Clone the repository
git clone https://github.com/felixsie19/MetaLNPs.git
cd MetaLNPs

##2. Create the conda environment
conda env create -f py310.yml
conda activate metalnps

Running Experiments
Few-shot vs Supervised Benchmark
python experiments/FewShotvsSupervisedBaseline/benchmark_compare.py

Active Learning
python experiments/ActiveLearning/run_AL.py

Inference on Custom Lipids
python experiments/OwnLipidsInference/run_lipid_inf.py

##Results

Results, plots, and intermediate outputs are stored within each experiment directory, including:

Performance metrics (CSV/XLSX)

Model checkpoints

Visualizations (e.g. UMAPs, performance curves)

Reproducibility

All experiments are seeded where applicable

Conda environment provided (py310.yml)

SLURM script included for HPC execution

Project Status

Research prototype / experimental codebase

This repository accompanies ongoing research and is intended for reproducibility and extension rather than production use.



License information to be added.

Contact

For questions or collaboration inquiries, please open an issue or contact the repository owner via GitHub.
