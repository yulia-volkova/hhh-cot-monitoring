#!/usr/bin/env python
"""
Main pipeline for VFT reproduction.
Run all stages: data prep, VFT/BCT training, RL, and evaluation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from src.data import (
    create_vft_bct_split,
    create_cued_dataset,
    create_paired_dataset,
    save_dataset,
    load_json_dataset,
    prepare_mmlu_example,
    load_mmlu_data,
)
from src.vft_dataset import construct_vft_dataset, construct_bct_dataset
from src.training import train_sft, train_grpo
from src.evaluation import evaluate_model, compute_mmlu_accuracy


# Cue types from the paper
VFT_BCT_CUES = ["black_square", "post_hoc"]  # Training cues
ALL_CUES = [
    "stanford_professor",
    "black_square",
    "wrong_few_shot",
    "post_hoc",
    "metadata",
    "validation_function",
    "unauthorized_access",
]
HELD_OUT_CUES = [c for c in ALL_CUES if c not in VFT_BCT_CUES]


def prepare_data(args):
    """Prepare all datasets."""
    print("=" * 50)
    print("Stage 1: Preparing Data")
    print("=" * 50)

    output_dir = Path(args.output_dir) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create splits
    vft_bct_examples, rl_examples, val_examples, test_examples = create_vft_bct_split(
        num_vft_bct=3352,
        num_rl=4210,
        num_val=1000,
        num_test=1000,
        seed=args.seed,
    )

    # Save raw splits
    save_dataset(vft_bct_examples, output_dir / "vft_bct_examples.json")
    save_dataset(rl_examples, output_dir / "rl_examples.json")
    save_dataset(val_examples, output_dir / "val_examples.json")
    save_dataset(test_examples, output_dir / "test_examples.json")

    # Create cued datasets for RL (90% cued, 10% uncued)
    rl_cued_data = create_cued_dataset(
        rl_examples,
        ALL_CUES,
        cued_fraction=0.9,
        seed=args.seed,
    )
    save_dataset(rl_cued_data, output_dir / "rl_cued_data.json")

    print(f"Saved data to {output_dir}")
    print(f"  VFT/BCT examples: {len(vft_bct_examples)}")
    print(f"  RL examples: {len(rl_examples)}")
    print(f"  Val examples: {len(val_examples)}")
    print(f"  Test examples: {len(test_examples)}")


def construct_vft_bct_datasets(args):
    """Construct VFT and BCT training datasets."""
    print("=" * 50)
    print("Stage 2: Constructing VFT/BCT Datasets")
    print("=" * 50)

    data_dir = Path(args.output_dir) / "data"
    output_dir = Path(args.output_dir) / "processed_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load examples
    vft_bct_examples = load_json_dataset(data_dir / "vft_bct_examples.json")

    # Create paired data
    paired_data = create_paired_dataset(vft_bct_examples, VFT_BCT_CUES, seed=args.seed)

    # Load models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Main model for response generation
    model = LLM(
        model=args.base_model,
        tensor_parallel_size=args.num_gpus,
        dtype="bfloat16",
    )

    # Judge model for verbalization checking
    # Using same model as judge since we don't have Claude API
    judge_tokenizer = tokenizer
    judge_model = model

    # Edit model for creating verbalizations
    # In the paper they use Claude 4 Sonnet, we'll use a larger Llama
    if args.edit_model:
        edit_tokenizer = AutoTokenizer.from_pretrained(args.edit_model)
        edit_model = LLM(
            model=args.edit_model,
            tensor_parallel_size=args.num_gpus,
            dtype="bfloat16",
        )
    else:
        edit_tokenizer = tokenizer
        edit_model = model

    # Construct VFT dataset
    print("Constructing VFT dataset...")
    vft_data = construct_vft_dataset(
        paired_data,
        model,
        tokenizer,
        judge_model,
        judge_tokenizer,
        edit_model,
        edit_tokenizer,
        batch_size=args.batch_size,
        output_dir=str(output_dir),
    )
    save_dataset(vft_data, output_dir / "vft_train_data.json")
    print(f"VFT dataset: {len(vft_data)} examples")

    # Construct BCT dataset
    if not args.skip_bct:
        print("Constructing BCT dataset...")
        bct_data = construct_bct_dataset(
            paired_data,
            model,
            tokenizer,
            batch_size=args.batch_size,
        )
        save_dataset(bct_data, output_dir / "bct_train_data.json")
        print(f"BCT dataset: {len(bct_data)} examples")
    else:
        print("Skipping BCT dataset construction (--skip_bct)")
        bct_data = []

    # Also add 10% uncued examples
    uncued_examples = load_json_dataset(data_dir / "vft_bct_examples.json")
    uncued_cued_data = create_cued_dataset(
        uncued_examples[:len(uncued_examples) // 10],
        VFT_BCT_CUES,
        cued_fraction=0.0,
        seed=args.seed,
    )

    # Format uncued for training
    uncued_train = []
    prompts = []
    for item in uncued_cued_data:
        from src.data import get_standard_messages
        messages = get_standard_messages(item["prompt"])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=1.0, max_tokens=1024)
    outputs = model.generate(prompts, sampling_params)

    for i, item in enumerate(uncued_cued_data):
        uncued_train.append({
            "prompt": item["prompt"],
            "response": outputs[i].outputs[0].text,
            "cue_type": None,
            "category": "uncued",
        })

    # Combine with VFT and BCT
    vft_data_with_uncued = vft_data + uncued_train
    save_dataset(vft_data_with_uncued, output_dir / "vft_train_data_full.json")
    print(f"Final VFT dataset: {len(vft_data_with_uncued)} examples")

    if not args.skip_bct:
        bct_data_with_uncued = bct_data + uncued_train
        save_dataset(bct_data_with_uncued, output_dir / "bct_train_data_full.json")
        print(f"Final BCT dataset: {len(bct_data_with_uncued)} examples")


def train_vft_bct(args):
    """Train VFT and BCT models with SFT."""
    print("=" * 50)
    print("Stage 3: Training VFT and BCT Models")
    print("=" * 50)

    data_dir = Path(args.output_dir) / "processed_data"
    models_dir = Path(args.output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    vft_data = load_json_dataset(data_dir / "vft_train_data_full.json")
    if not args.skip_bct:
        bct_data = load_json_dataset(data_dir / "bct_train_data_full.json")
    val_data = load_json_dataset(Path(args.output_dir) / "data" / "val_examples.json")

    # Create validation set in training format
    val_train_format = []
    for item in val_data[:100]:  # Small val set
        val_train_format.append({
            "prompt": f"{item['question']}\nA) {item['choices'][0]}\nB) {item['choices'][1]}\nC) {item['choices'][2]}\nD) {item['choices'][3]}",
            "response": f"Let me think about this step by step.\n\nAnswer: {item['correct_answer']}",
        })

    # Train VFT
    print("\nTraining VFT model...")
    train_sft(
        model_name=args.base_model,
        train_data=vft_data,
        val_data=val_train_format,
        output_dir=str(models_dir / "vft"),
        num_epochs=1,
        batch_size=256,
        learning_rate=1e-5,
        warmup_steps=10,
        gradient_accumulation_steps=args.grad_accum,
        use_lora=args.use_lora,
        wandb_project=args.wandb_project,
        run_name="vft-sft",
    )

    # Train BCT
    if not args.skip_bct:
        print("\nTraining BCT model...")
        train_sft(
            model_name=args.base_model,
            train_data=bct_data,
            val_data=val_train_format,
            output_dir=str(models_dir / "bct"),
            num_epochs=1,
            batch_size=256,
            learning_rate=1e-5,
            warmup_steps=10,
            gradient_accumulation_steps=args.grad_accum,
            use_lora=args.use_lora,
            wandb_project=args.wandb_project,
            run_name="bct-sft",
        )
    else:
        print("\nSkipping BCT training (--skip_bct)")

    print("VFT training complete!" if args.skip_bct else "VFT and BCT training complete!")


def train_rl(args):
    """Train with RL for reward hacking amplification."""
    print("=" * 50)
    print("Stage 4: RL Training (Reward Hacking Amplification)")
    print("=" * 50)

    data_dir = Path(args.output_dir) / "data"
    models_dir = Path(args.output_dir) / "models"

    # Load RL data
    rl_data = load_json_dataset(data_dir / "rl_cued_data.json")

    # Model paths
    model_configs = [
        ("baseline", args.base_model),
        ("vft", str(models_dir / "vft" / "checkpoint-70")),  # VFT checkpoint 70
        ("bct", str(models_dir / "bct" / "checkpoint-50")),  # BCT checkpoint 50
    ]

    for name, model_path in model_configs:
        if not os.path.exists(model_path) and name != "baseline":
            print(f"Skipping {name} - model not found at {model_path}")
            continue

        print(f"\nTraining {name} with RL...")
        train_grpo(
            model_path=model_path,
            train_data=rl_data,
            output_dir=str(models_dir / f"{name}_rl"),
            num_steps=150,
            batch_size=16,  # 4 GPUs * 4 grad_accum * 1 per_device = 16
            num_rollouts=4,
            learning_rate=1e-5,
            warmup_steps=10,
            kl_coef=0.001,
            gradient_accumulation_steps=args.grad_accum,
            wandb_project=args.wandb_project,
            run_name=f"{name}-grpo",
        )

    print("RL training complete!")


def evaluate(args):
    """Evaluate all models."""
    print("=" * 50)
    print("Stage 5: Evaluation")
    print("=" * 50)

    data_dir = Path(args.output_dir) / "data"
    models_dir = Path(args.output_dir) / "models"
    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_examples = load_json_dataset(data_dir / "test_examples.json")

    # Load tokenizers and models
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Models to evaluate
    model_configs = [
        ("baseline", args.base_model),
        ("vft", str(models_dir / "vft" / "checkpoint-70")),
        ("bct", str(models_dir / "bct" / "checkpoint-50")),
        ("baseline_rl", str(models_dir / "baseline_rl" / "checkpoint-100")),
        ("vft_rl", str(models_dir / "vft_rl" / "checkpoint-100")),
        ("bct_rl", str(models_dir / "bct_rl" / "checkpoint-100")),
    ]

    all_results = {}

    for name, model_path in model_configs:
        if not os.path.exists(model_path) and name != "baseline":
            print(f"Skipping {name} - model not found")
            continue

        print(f"\nEvaluating {name}...")

        # Load model
        model = LLM(
            model=model_path,
            tensor_parallel_size=args.num_gpus,
            dtype="bfloat16",
        )

        # Judge model (same model for simplicity)
        judge_model = model
        judge_tokenizer = tokenizer

        # Evaluate on held-out cues
        results = evaluate_model(
            model,
            tokenizer,
            judge_model,
            judge_tokenizer,
            test_examples,
            HELD_OUT_CUES,
            output_path=str(results_dir / f"{name}_held_out.json"),
            batch_size=args.batch_size,
        )

        # Also evaluate on training cues
        results_train = evaluate_model(
            model,
            tokenizer,
            judge_model,
            judge_tokenizer,
            test_examples,
            VFT_BCT_CUES,
            output_path=str(results_dir / f"{name}_train_cues.json"),
            batch_size=args.batch_size,
        )

        # MMLU accuracy
        mmlu_acc = compute_mmlu_accuracy(model, tokenizer, test_examples, batch_size=args.batch_size)

        all_results[name] = {
            "held_out_cues": results["overall"],
            "train_cues": results_train["overall"],
            "mmlu_accuracy": mmlu_acc,
        }

        print(f"{name} results:")
        print(f"  Held-out ECR: {results['overall']['ecr']:.2%}")
        print(f"  Held-out Verbalization Rate: {results['overall']['verbalization_rate']:.2%}")
        print(f"  Held-out Cue Influence Rate: {results['overall']['cue_influence_rate']:.2%}")
        print(f"  MMLU Accuracy: {mmlu_acc:.2%}")

    # Save summary
    with open(results_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'ECR':>8} {'VerbRate':>10} {'CueInfl':>10} {'MMLU':>8}")
    print("-" * 80)

    for name, results in all_results.items():
        ecr = results["held_out_cues"]["ecr"]
        vr = results["held_out_cues"]["verbalization_rate"]
        cir = results["held_out_cues"]["cue_influence_rate"]
        acc = results["mmlu_accuracy"]
        print(f"{name:<15} {ecr:>7.1%} {vr:>9.1%} {cir:>9.1%} {acc:>7.1%}")


def main():
    parser = argparse.ArgumentParser(description="VFT Reproduction Pipeline")

    # General arguments
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for all artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs for tensor parallelism")

    # Model arguments
    parser.add_argument("--base_model", type=str,
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Base model to use")
    parser.add_argument("--edit_model", type=str, default=None,
                       help="Model for editing verbalizations (default: same as base)")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for data processing")
    parser.add_argument("--grad_accum", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for fine-tuning")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="vft-reproduction",
                       help="W&B project name")

    # Stages to run
    parser.add_argument("--stage", type=str, default="all",
                       choices=["all", "data", "construct", "train_sft", "train_rl", "eval"],
                       help="Which stage to run")
    parser.add_argument("--skip_bct", action="store_true",
                       help="Skip BCT dataset construction (only generate VFT data)")

    args = parser.parse_args()

    # Resolve base_model to absolute path if it's a relative path
    if args.base_model and not args.base_model.startswith('/') and '/' in args.base_model:
        # Looks like a relative path (e.g., ../models/...), resolve it
        resolved_path = Path(args.base_model).resolve()
        args.base_model = str(resolved_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {resolved_path}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run requested stages
    if args.stage == "all":
        prepare_data(args)
        construct_vft_bct_datasets(args)
        train_vft_bct(args)
        train_rl(args)
        evaluate(args)
    elif args.stage == "data":
        prepare_data(args)
    elif args.stage == "construct":
        construct_vft_bct_datasets(args)
    elif args.stage == "train_sft":
        train_vft_bct(args)
    elif args.stage == "train_rl":
        train_rl(args)
    elif args.stage == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
