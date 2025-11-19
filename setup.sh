#!/bin/bash
# Setup script for VFT reproduction using uv

set -e

echo "Setting up VFT reproduction environment..."

# Load CUDA module
module purge
module load cudatoolkit/12.8

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment with uv
echo "Creating virtual environment..."
uv venv .venv --python 3.10

# Activate environment
source .venv/bin/activate

# Install dependencies with uv
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Install flash-attention (optional, for faster training)
echo "Installing flash-attention..."
uv pip install flash-attn --no-build-isolation || echo "flash-attn installation failed (optional)"

# Create directories
mkdir -p logs output data

# Generate individual SLURM stage scripts
bash scripts/slurm_stages.sh

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To login to HuggingFace (needed for Llama):"
echo "  huggingface-cli login"
echo ""
echo "To run the full pipeline:"
echo "  sbatch scripts/slurm_full_pipeline.sh"
