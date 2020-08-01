#!/bin/bash
#SBATCH --time 5:00:00
#SBATCH --out job.slurm.out
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1 
#SBATCH --mem=20G 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=15 

python3 AE_selection.py
