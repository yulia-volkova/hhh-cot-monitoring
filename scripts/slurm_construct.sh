#!/bin/bash
#SBATCH --job-name=vft-construct
#SBATCH --output=logs/vft-construct-%j.out
#SBATCH --error=logs/vft-construct-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=1:00:00

module purge
module load cudatoolkit/12.8

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p logs

python scripts/run_pipeline.py \
    --output_dir ./output \
    --base_model ../../models/Llama-3.1-8B-Instruct \
    --num_gpus 8 \
    --batch_size 256 \
    --stage construct \
    --skip_bct
