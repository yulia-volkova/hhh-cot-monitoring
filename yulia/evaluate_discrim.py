"""Evaluate reasoning models on the Anthropic discrim-eval dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from models import ReasoningTransformersClient, ReasoningVLLMClient, GenerationResult
from utils import set_seed

from yulia.dataset import (
    DiscrimExample,
    format_prompt_for_reasoning,
    load_discrim_eval,
)


def extract_yes_no(text: str) -> Optional[str]:
    text_lower = text.lower()
    
    # Look after </think> tag if present
    if "</think>" in text_lower:
        text_lower = text_lower.split("</think>", 1)[1]
    
    # Simple heuristic: check for yes/no in the response
    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    
    if has_yes and not has_no:
        return "Yes"
    elif has_no and not has_yes:
        return "No"
    
    # If both or neither, try to find the first occurrence
    yes_idx = text_lower.find("yes")
    no_idx = text_lower.find("no")
    
    if yes_idx >= 0 and (no_idx < 0 or yes_idx < no_idx):
        return "Yes"
    elif no_idx >= 0:
        return "No"
    
    return None


def evaluate_batch(
    client,
    examples: List[DiscrimExample],
    *,
    batch_size: int = 1,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    enable_reasoning: bool = True,
) -> List[Dict[str, Any]]:

    results = []
    
    for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
        batch = examples[i : i + batch_size]
        prompts = [format_prompt_for_reasoning(ex, enable_reasoning=enable_reasoning) for ex in batch]
        
        # Generate responses
        generation_results = client.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        # Process results
        for example, gen_result in zip(batch, generation_results):
            decision = extract_yes_no(gen_result.text)
            
            result = {
                "filled_template": example.filled_template,
                "decision_question_id": example.decision_question_id,
                "target_group": example.target_group,
                "age": example.age,
                "gender": example.gender,
                "race": example.race,
                "fill_type": example.fill_type,
                # Model outputs
                "response": gen_result.text,
                "reasoning": gen_result.reasoning,
                "decision": decision,
                "prompt": gen_result.prompt,
            }
            results.append(result)
    
    return results


def _save_results(results: List[Dict[str, Any]], metadata: Dict[str, Any], run_dir: Path) -> Path:
    """Save evaluation results as JSONL."""
    # Save as JSONL
    jsonl_path = run_dir / "discrim_eval_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved JSONL results to {jsonl_path}")
    
    # Save metadata
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {metadata_path}")
    
    return jsonl_path


def run_evaluation(
    model_name: str,
    output_dir: Path,
    *,
    config: str = "explicit",
    max_examples: Optional[int] = None,
    **kwargs,
) -> Path:
    """
    Run evaluation on discrim-eval dataset.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save results
        config: Dataset config ("explicit" or "implicit")
        max_examples: Limit number of examples (None for all)
        **kwargs: Additional options:
            - split: Dataset split (default: "train")
            - seed: Random seed (default: 42)
            - max_new_tokens: Max tokens to generate (default: 2048)
            - temperature: Sampling temperature (default: 0.7)
            - backend: "transformers" or "vllm" (default: "vllm")
            - batch_size: Batch size for inference (default: 128)
            - enable_reasoning: Show reasoning/CoT (default: True)
            - run_name: Custom run name (default: timestamp)
    
    Returns:
        Path to the saved results file
    """
    # Extract kwargs with defaults
    split = kwargs.get("split", "train")
    seed = kwargs.get("seed", 42)
    max_new_tokens = kwargs.get("max_new_tokens", 2048)
    temperature = kwargs.get("temperature", 0.7)
    backend = kwargs.get("backend", "vllm")
    batch_size = kwargs.get("batch_size", 128)
    enable_reasoning = kwargs.get("enable_reasoning", True)
    run_name = kwargs.get("run_name", None)
    
    set_seed(seed)
    
    # Create output directory
    run_dir = output_dir / (run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading discrim-eval dataset (config={config}, split={split})...")
    examples = load_discrim_eval(split=split, config=config, max_examples=max_examples)
    print(f"Loaded {len(examples)} examples")
    
    # Initialize model client
    print(f"Initializing {backend} client with model {model_name}...")
    if backend == "transformers":
        client = ReasoningTransformersClient(model_name=model_name)
    elif backend == "vllm":
        client = ReasoningVLLMClient(model_name=model_name, seed=seed)
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Choose 'transformers' or 'vllm'.")
    
    # Run evaluation
    print(f"Running evaluation (reasoning: {'ON' if enable_reasoning else 'OFF'})...")
    results = evaluate_batch(
        client,
        examples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        enable_reasoning=enable_reasoning,
    )
    
    # Prepare and save results
    metadata = {
        "model_name": model_name,
        "backend": backend,
        "config": config,
        "split": split,
        "max_examples": max_examples,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "batch_size": batch_size,
        "enable_reasoning": enable_reasoning,
        "timestamp": datetime.now().isoformat(),
        "num_examples": len(results),
    }
    
    return _save_results(results, metadata, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reasoning models on Anthropic discrim-eval"
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model name to evaluate",
    )
    parser.add_argument(
        "--output",
        default="results/discrim_eval",
        help="Directory for saving results",
    )
    parser.add_argument(
        "--config",
        choices=["explicit", "implicit"],
        default="explicit",
        help="Dataset configuration",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Backend to use for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference (128 for A100 80GB)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name for output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        args.model,
        Path(args.output),
        config=args.config,
        max_examples=args.max_examples,
        split=args.split,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        backend=args.backend,
        batch_size=args.batch_size,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()

