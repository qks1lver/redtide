#!/usr/bin/env bash

#SBATCH -J growth
#SBATCH -p DPB
#SBATCH -c 24
#SBATCH --mem=4000

module purge; module load Python/3.6.0

srun python run.py -d -v
