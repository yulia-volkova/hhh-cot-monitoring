#!/bin/bash
#SBATCH --job-name=vft-eval
#SBATCH --output=logs/vft-eval-%j.out
#SBATCH --error=logs/vft-eval-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=8:00:00

module purge
module load cudatoolkit/12.8

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
mkdir -p logs

python scripts/run_pipeline.py \
    --output_dir ./output \
    --base_model ../../models/Llama-3.1-8B-Instruct \
    --num_gpus 2 \
    --stage eval
