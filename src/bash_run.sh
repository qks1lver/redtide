#!/usr/bin/env bash

#SBATCH -J growth
#SBATCH -p DPB
#SBATCH -c 8
#SBATCH --mem=8000

module purge; module load Python/3.6.0

srun python run.py "$@"
