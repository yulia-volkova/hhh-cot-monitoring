#!/bin/bash
#SBATCH --job-name=vft-construct
#SBATCH --output=logs/vft-construct-%j.out
#SBATCH --error=logs/vft-construct-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=1:00:00

module purge
module load cudatoolkit/12.8

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
mkdir -p logs

python scripts/run_pipeline.py \
    --output_dir ./output \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --num_gpus 2 \
    --batch_size 32 \
    --stage construct
