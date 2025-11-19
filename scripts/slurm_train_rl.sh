#!/bin/bash
#SBATCH --job-name=vft-rl
#SBATCH --output=logs/vft-rl-%j.out
#SBATCH --error=logs/vft-rl-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00

module purge
module load cudatoolkit/12.8

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="vft-reproduction"
mkdir -p logs

python scripts/run_pipeline.py \
    --output_dir ./output \
    --base_model ../models/Llama-3.1-8B-Instruct \
    --num_gpus 4 \
    --grad_accum 4 \
    --stage train_rl
