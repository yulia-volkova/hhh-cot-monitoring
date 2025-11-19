#!/bin/bash
#SBATCH --job-name=vft-data
#SBATCH --output=logs/vft-data-%j.out
#SBATCH --error=logs/vft-data-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

module purge

source .venv/bin/activate

mkdir -p logs

python scripts/run_pipeline.py \
    --output_dir ./output \
    --stage data
