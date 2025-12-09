#!/bin/bash
#SBATCH --job-name=my_dataset_job
#SBATCH --account=cis220031p
#SBATCH --partition=RM
#SBATCH --qos=ft
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ulimit -v unlimited
free -h
vmstat -s

python encode_dataset.py

