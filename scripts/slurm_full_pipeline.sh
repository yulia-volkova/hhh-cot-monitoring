#!/bin/bash
#SBATCH --job-name=vft-full
#SBATCH --output=logs/vft-full-%j.out
#SBATCH --error=logs/vft-full-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00

# VFT Reproduction - Full Pipeline
# This runs all stages sequentially on a single node with 4 GPUs

set -e  # Exit on error

# Load modules (adjust for your cluster)
module purge
module load cudatoolkit/12.8

# Activate environment
source .venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="vft-reproduction"

# Create logs directory
mkdir -p logs

# Run full pipeline
python scripts/run_pipeline.py \
    --output_dir ./output \
    --base_model ../models/Llama-3.1-8B-Instruct \
    --num_gpus 4 \
    --batch_size 64 \
    --grad_accum 4 \
    --seed 42 \
    --stage all

echo "Full pipeline complete!"
