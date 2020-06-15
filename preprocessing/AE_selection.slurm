#!/bin/bash
#SBATCH --time 5:00:00
#SBATCH --out job.slurm.out
#SBATCH --mem=10G 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=15 

python3 AE_selection.py
