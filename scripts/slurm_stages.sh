#!/bin/bash
# Individual SLURM scripts for running stages separately
# Useful for debugging or resuming from a specific stage

# =============================================================================
# Stage 1: Data Preparation (CPU only, quick)
# =============================================================================
cat > scripts/slurm_data.sh << 'EOF'
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
EOF

# =============================================================================
# Stage 2: Construct VFT/BCT Datasets (needs GPU for generation)
# =============================================================================
cat > scripts/slurm_construct.sh << 'EOF'
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
    --batch_size 128 \
    --stage construct
EOF

# =============================================================================
# Stage 3: Train VFT/BCT with SFT
# =============================================================================
cat > scripts/slurm_train_sft.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=vft-sft
#SBATCH --output=logs/vft-sft-%j.out
#SBATCH --error=logs/vft-sft-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00

module purge
module load cudatoolkit/12.8

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="vft-reproduction"
mkdir -p logs

python scripts/run_pipeline.py \
    --output_dir ./output \
    --base_model ../../models/Llama-3.1-8B-Instruct \
    --num_gpus 4 \
    --grad_accum 4 \
    --stage train_sft
EOF

# =============================================================================
# Stage 4: Train with RL (GRPO)
# =============================================================================
cat > scripts/slurm_train_rl.sh << 'EOF'
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
    --base_model ../../models/Llama-3.1-8B-Instruct \
    --num_gpus 4 \
    --grad_accum 4 \
    --stage train_rl
EOF

# =============================================================================
# Stage 5: Evaluation
# =============================================================================
cat > scripts/slurm_eval.sh << 'EOF'
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
EOF

echo "Individual stage scripts created in scripts/"
echo "Run with: sbatch scripts/slurm_<stage>.sh"
