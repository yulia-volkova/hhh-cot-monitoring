"""Few-shot cue evaluation pipeline for reasoning models on MMLU."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm

from mmlu import build_prompt, load_mmlu, to_mmlu_example
from models import GenerationResult, ReasoningTransformersClient, ReasoningVLLMClient
from utils import extract_choice_from_output, set_seed, write_csv


def _evaluate_subject(
    client,
    *,
    subject: str,
    output_dir: Path,
    max_examples: Optional[int],
    max_new_tokens: int,
    temperature: float,
) -> None:
    test_split = load_mmlu(split="test", subjects=[subject])
    dev_examples = load_mmlu(split="dev", subjects=[subject])

    few_shots = [to_mmlu_example(dev_examples[i]) for i in range(len(dev_examples))]

    jsonl_path = output_dir / f"{subject}_results.jsonl"
    records: List[Dict[str, object]] = []

    for idx, raw_item in enumerate(tqdm(test_split, desc=f"{subject} examples")):
        if max_examples is not None and idx >= max_examples:
            break

        eval_example = to_mmlu_example(raw_item)

        baseline_prompt = build_prompt(
            eval_example,
            few_shot_examples=few_shots,
            mark_fewshot_correct=False,
            cue_choice=None,
        )
        baseline_result = client.generate(
            baseline_prompt["messages"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        baseline_choice = baseline_result.choice or extract_choice_from_output(baseline_result.text)

        incorrect_choices = [
            i
            for i in range(len(eval_example.choices))
            if eval_example.choices[i] and i != eval_example.answer_index
        ]
        cue_choice = random.choice(incorrect_choices) if incorrect_choices else None

        cue_prompt = build_prompt(
            eval_example,
            few_shot_examples=few_shots,
            mark_fewshot_correct=True,
            cue_choice=cue_choice,
        )
        cue_result = client.generate(
            cue_prompt["messages"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        cue_choice_letter = cue_result.choice or extract_choice_from_output(cue_result.text)

        record = {
            "id": eval_example.uid or f"{subject}_{idx}",
            "subject": subject,
            "question": eval_example.question,
            "choices": list(eval_example.choices),
            "correct_answer": eval_example.answer_label,
            "baseline_choice": baseline_choice,
            "baseline_text": baseline_result.text,
            "baseline_reasoning": baseline_result.reasoning,
            "cue_choice": chr(ord("A") + cue_choice) if cue_choice is not None else None,
            "cue_choice_model": cue_choice_letter,
            "cue_text": cue_result.text,
            "cue_reasoning": cue_result.reasoning,
            "baseline_metadata": {
                "prompt": baseline_result.prompt,
            },
            "cue_metadata": {
                "prompt": cue_result.prompt,
            },
        }
        records.append(record)

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
) -> None:
    set_seed(seed)
    run_dir = output_dir / (run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if backend == "transformers":
        client = ReasoningTransformersClient(model_name=model_name)
    elif backend == "vllm":
        client = ReasoningVLLMClient(model_name=model_name)
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
    )


if __name__ == "__main__":
    main()
