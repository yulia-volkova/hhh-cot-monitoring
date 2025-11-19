# VFT Reproduction

Reproduction of "Teaching Models to Verbalize Reward Hacking in Chain-of-Thought Reasoning" (Turpin et al., 2025).

## Overview

This repository implements Verbalization Fine-Tuning (VFT), a method to train language models to explicitly acknowledge when they are exploiting reward hacks. The key insight is that instead of trying to prevent reward hacking, we can make it detectable by training models to verbalize when they are influenced by prompt cues.

## Key Results (Paper)

| Model | ECR ↓ | Verbalization Rate ↑ | Cue Influence Rate |
|-------|-------|---------------------|-------------------|
| Baseline → RL | 88% | 11% | ~100% |
| BCT → RL | 99% | 1% | ~100% |
| **VFT → RL** | **6%** | **94%** | ~100% |

## Setup

```bash
# Clone and setup
bash setup.sh

# Login to HuggingFace (needed for Llama)
huggingface-cli login
```

## Running the Pipeline

### Full Pipeline (Recommended)

```bash
sbatch scripts/slurm_full_pipeline.sh
```

This runs all 5 stages:
1. Data preparation
2. VFT/BCT dataset construction
3. SFT training
4. RL training (GRPO)
5. Evaluation

### Individual Stages

```bash
# Generate stage scripts
bash scripts/slurm_stages.sh

# Run stages separately
sbatch scripts/slurm_data.sh        # ~1 hour
sbatch scripts/slurm_construct.sh   # ~8 hours
sbatch scripts/slurm_train_sft.sh   # ~12 hours
sbatch scripts/slurm_train_rl.sh    # ~24 hours
sbatch scripts/slurm_eval.sh        # ~8 hours
```

## Method Summary

### 1. Prompt Cues
Seven types of cues that bias models toward incorrect answers:
- Stanford professor
- Black square
- Wrong few-shot
- Post-hoc
- Metadata
- Validation function
- Unauthorized access

### 2. VFT Dataset Construction
1. Generate responses to cued and uncued prompts
2. Categorize as:
   - **Faithful non-switch**: Answer doesn't change → use baseline reasoning
   - **Faithful switch**: Model verbalizes → keep as-is
   - **Unfaithful switch**: Model doesn't verbalize → edit to add verbalization

### 3. Training
- **VFT**: Fine-tune to verbalize cue influence
- **BCT** (baseline): Fine-tune to ignore cues
- **RL**: Train with GRPO on flawed reward that rewards cue-aligned wrong answers

### 4. Metrics
- **Cue Influence Rate**: How often model follows the cue
- **Verbalization Rate**: How often model admits to following the cue
- **ECR (Effective Cue Influence Rate)**: (1 - VR) × CIR = undetected hacks

## Project Structure

```
VFT/
├── src/
│   ├── cues.py           # Prompt cue implementations
│   ├── data.py           # Data loading and processing
│   ├── vft_dataset.py    # VFT dataset construction
│   ├── training.py       # SFT and GRPO training
│   └── evaluation.py     # Metrics computation
├── scripts/
│   ├── run_pipeline.py   # Main pipeline script
│   └── slurm_*.sh        # SLURM job scripts
├── configs/
│   └── default.yaml      # Hyperparameters
├── requirements.txt
└── README.md
```

## Differences from Paper

1. **Judge model**: Uses Llama instead of Claude 3.5 Sonnet
2. **Edit model**: Uses Llama instead of Claude 4 Sonnet
3. **Dataset**: Uses MMLU "all" split instead of specific subjects

## Hyperparameters (from Paper Appendix B)

### SFT
- Steps: 70 (VFT), 50 (BCT)
- Batch size: 256
- Learning rate: 1e-5
- Warmup: 10 steps
- Scheduler: Cosine decay

### GRPO
- Steps: 150
- Batch size: 256 prompts × 8 rollouts
- Learning rate: 1e-5
- KL coefficient: 0.001

## Citation

```bibtex
@article{turpin2025verbalize,
  title={Teaching Models to Verbalize Reward Hacking in Chain-of-Thought Reasoning},
  author={Turpin, Miles and Arditi, Andy and Li, Marvin and Benton, Joe and Michael, Julian},
  journal={arXiv preprint arXiv:2506.22777},
  year={2025}
}
```

## License

MIT
