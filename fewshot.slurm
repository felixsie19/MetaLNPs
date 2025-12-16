#!/bin/bash
#SBATCH --job-name=fewshot
#SBATCH --output=slurm-%j.out       # output log (job ID will be inserted)
#SBATCH --error=slurm-%j.err        # error log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20         # adjust for your script
#SBATCH --mem=120G                   # memory limit
#SBATCH --time=80:00:00             # max run time
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --mail-type=END,FAIL        # email on job end/fail
#SBATCH --mail-user=felsiph@cup.uni-muenchen.de


conda activate py310


cd /home/akm/Felix_ML/Lasse_lipids/FewShotLNPs

# === Run the Python module as a package ===
python -m models.few_shot
