"""Few-shot cue evaluation pipeline for reasoning models on MMLU."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm

from mmlu import DEFAULT_CUE_STYLE, SUPPORTED_CUE_STYLES, build_prompt, load_mmlu, to_mmlu_example
from models import ReasoningTransformersClient, ReasoningVLLMClient
from utils import extract_choice_from_output, set_seed, write_csv


def _evaluate_subject(
    client,
    *,
    subject: str,
    output_dir: Path,
    max_examples: Optional[int],
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    cue_style: str,
) -> None:
    test_split = load_mmlu(split="test", subjects=[subject])
    dev_examples = load_mmlu(split="dev", subjects=[subject])

    few_shots = [to_mmlu_example(dev_examples[i]) for i in range(len(dev_examples))]

    jsonl_path = output_dir / f"{subject}_results.jsonl"
    records: List[Dict[str, object]] = []

    batch: List[Dict[str, object]] = []

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return

        baseline_prompts = [item["baseline_prompt"]["prompt"] for item in batch]
        baseline_results = client.generate_batch(
            baseline_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        cue_prompts = []
        for item, baseline_result in zip(batch, baseline_results):
            eval_example = item["eval_example"]
            baseline_choice = baseline_result.choice or extract_choice_from_output(baseline_result.text)

            incorrect_choices = [
                i
                for i in range(len(eval_example.choices))
                if eval_example.choices[i] and i != eval_example.answer_index
            ]

            baseline_idx: Optional[int] = None
            if baseline_choice and len(baseline_choice) == 1 and "A" <= baseline_choice <= "Z":
                baseline_idx = ord(baseline_choice) - ord("A")
            cue_candidates = [i for i in incorrect_choices if baseline_idx is None or i != baseline_idx]
            if not cue_candidates:
                cue_candidates = incorrect_choices
            cue_choice = random.choice(cue_candidates) if cue_candidates else None

            cue_prompt = build_prompt(
                eval_example,
                few_shot_examples=few_shots,
                mark_fewshot_correct=True,
                cue_choice=cue_choice,
                cue_style=cue_style,
            )

            item.update(
                {
                    "baseline_result": baseline_result,
                    "baseline_choice": baseline_choice,
                    "cue_choice_index": cue_choice,
                    "cue_prompt": cue_prompt,
                }
            )
            cue_prompts.append(cue_prompt["prompt"])

        cue_results = client.generate_batch(
            cue_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        for item, cue_result in zip(batch, cue_results):
            eval_example = item["eval_example"]
            cue_choice = item["cue_choice_index"]
            cue_choice_letter = cue_result.choice or extract_choice_from_output(cue_result.text)

            record = {
                "id": eval_example.uid or f"{subject}_{item['idx']}",
                "subject": subject,
                "question": eval_example.question,
                "choices": list(eval_example.choices),
                "correct_answer": eval_example.answer_label,
                "baseline_choice": item["baseline_choice"],
                "baseline_text": item["baseline_result"].text,
                "baseline_reasoning": item["baseline_result"].reasoning,
                "cue_choice": chr(ord("A") + cue_choice) if cue_choice is not None else None,
                "cue_choice_model": cue_choice_letter,
                "cue_text": cue_result.text,
                "cue_reasoning": cue_result.reasoning,
                "baseline_metadata": {
                    "prompt": item["baseline_result"].prompt,
                },
                "cue_metadata": {
                    "prompt": cue_result.prompt,
                    "cue_style": cue_style,
                },
            }
            records.append(record)

        batch = []

    for idx, raw_item in enumerate(tqdm(test_split, desc=f"{subject} examples")):
        if max_examples is not None and idx >= max_examples:
            break

        eval_example = to_mmlu_example(raw_item)

        baseline_prompt = build_prompt(
            eval_example,
            few_shot_examples=few_shots,
            mark_fewshot_correct=False,
            cue_choice=None,
            cue_style=cue_style,
        )
        batch.append(
            {
                "idx": idx,
                "eval_example": eval_example,
                "baseline_prompt": baseline_prompt,
            }
        )

        if len(batch) >= batch_size:
            flush_batch()

    flush_batch()

    if records:
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        csv_rows: Iterable[Dict[str, object]] = (
            {
                "id": r["id"],
                "subject": r["subject"],
                "correct": r["correct_answer"],
                "baseline": r["baseline_choice"],
                "cue_target": r["cue_choice"],
                "cue_model": r["cue_choice_model"],
            }
            for r in records
        )
        write_csv(
            output_dir / f"{subject}_summary.csv",
            csv_rows,
            fieldnames=["id", "subject", "correct", "baseline", "cue_target", "cue_model"],
        )


def run_evaluation(
    *,
    subjects: Sequence[str],
    model_name: str,
    output_dir: Path,
    max_examples: Optional[int] = None,
    seed: int = 42,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    run_name: Optional[str] = None,
    backend: str = "transformers",
    batch_size: int = 1,
    cue_style: str = DEFAULT_CUE_STYLE,
) -> None:
    set_seed(seed)
    run_dir = output_dir / (run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if cue_style not in SUPPORTED_CUE_STYLES:
        raise ValueError(
            f"Unsupported cue_style '{cue_style}'. Expected one of {sorted(SUPPORTED_CUE_STYLES)}."
        )

    if backend == "transformers":
        client = ReasoningTransformersClient(model_name=model_name)
    elif backend == "vllm":
        client = ReasoningVLLMClient(model_name=model_name, seed=seed)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported backend '{backend}'. Choose 'transformers' or 'vllm'.")

    for subject in tqdm(subjects, desc="Subjects"):
        _evaluate_subject(
            client,
            subject=subject,
            output_dir=run_dir,
            max_examples=max_examples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            cue_style=cue_style,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot cue evaluation for reasoning models")
    parser.add_argument("subjects", nargs="+", help="One or more MMLU subjects (e.g., high_school_physics)")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output", default="results", help="Directory for saving results")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--run-name", default=None, help="Optional run name subdirectory")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts to evaluate in parallel")
    parser.add_argument(
        "--cue-style",
        choices=sorted(SUPPORTED_CUE_STYLES),
        default=DEFAULT_CUE_STYLE,
        help="Select which cue formulation to inject during the second pass",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        subjects=args.subjects,
        model_name=args.model,
        output_dir=Path(args.output),
        max_examples=args.max_examples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        run_name=args.run_name,
        backend=args.backend,
        batch_size=args.batch_size,
        cue_style=args.cue_style,
    )


if __name__ == "__main__":
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
