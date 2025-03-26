#!/bin/bash
#SBATCH --job-name=gps_array
#SBATCH --array=0-17
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G

CONFIGS=(
  "100000 2"
  "100000 4"
  "100000 8"
  "100000 16"
  "100000 32"
  "250000 2"
  "250000 4"
  "250000 8"
  "250000 16"
  "250000 32"
  "500000 2"
  "500000 4"
  "500000 8"
  "500000 16"
  "1000000 2"
  "1000000 4"
  "1000000 8"
  "1000000 12"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read CHUNK WORKERS <<< "$CONFIG"

python test.py --chunk_size $CHUNK --n_workers $WORKERS