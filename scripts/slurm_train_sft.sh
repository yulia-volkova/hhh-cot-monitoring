#!/bin/bash
#SBATCH --job-name=vft-sft
#SBATCH --output=logs/vft-sft-%j.out
#SBATCH --error=logs/vft-sft-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=1:00:00

module purge
module load cudatoolkit/12.8

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="vft-reproduction"
mkdir -p logs

python scripts/run_pipeline.py \
    --output_dir ./output \
    --base_model ../../models/Llama-3.1-8B-Instruct \
    --num_gpus 8 \
    --grad_accum 4 \
    --stage train_sft
