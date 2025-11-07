"""Utilities for loading MMLU data and constructing few-shot prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset, load_dataset

OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
VISUAL_CUE = "✓"

MMLU_STEM_SUBJECTS: Tuple[str, ...] = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "machine_learning",
    "medical_genetics",
    "professional_medicine",
    "virology",
)


@dataclass(frozen=True)
class MMLUExample:
    """Normalized representation of a single MMLU item."""

    question: str
    choices: Tuple[str, str, str, str]
    answer_index: int
    subject: str
    uid: Optional[str] = None

    @property
    def answer_label(self) -> str:
        return OPTION_LABELS[self.answer_index]


def _normalize_answer(answer: Any) -> int:
    """Map raw answer labels to integer indices."""
    if isinstance(answer, str):
        label = answer.strip().upper()
        if not label:
            raise ValueError("Empty answer label encountered.")
        return OPTION_LABELS.index(label[0])
    if isinstance(answer, int):
        if 0 <= answer < len(OPTION_LABELS):
            return answer
        raise ValueError(f"Answer index {answer} out of range.")
    raise TypeError(f"Unsupported answer type: {type(answer)!r}")


def to_mmlu_example(item: Dict[str, Any], *, uid: Optional[str] = None) -> MMLUExample:
    """Convert a raw dataset row into an :class:`MMLUExample`."""
    choices = item.get("choices")
    if not isinstance(choices, Iterable):
        raise TypeError("Each MMLU row must contain an iterable 'choices' field.")
    choices_list = list(choices)
    if len(choices_list) != len(OPTION_LABELS):
        raise ValueError("Each MMLU question must have exactly four answer choices.")

    return MMLUExample(
        question=str(item["question"]),
        choices=tuple(str(choice) for choice in choices_list),
        answer_index=_normalize_answer(item["answer"]),
        subject=str(item.get("subject", "")),
        uid=uid or item.get("id"),
    )


def load_mmlu(
    split: str = "dev",
    *,
    subjects: Optional[Sequence[str]] = MMLU_STEM_SUBJECTS,
    cache_dir: Optional[str] = None,
    load_config: str = "all",
) -> Dataset:
    """
    Load the `cais/mmlu` dataset split and optionally filter to a subject subset.
    """
    dataset = load_dataset(
        "cais/mmlu",
        load_config,
        split=split,
        cache_dir=cache_dir,
    )

    if subjects:
        subject_set = set(subjects)
        dataset = dataset.filter(lambda ex: ex.get("subject") in subject_set)

    return dataset


def dataset_to_examples(dataset: Dataset) -> List[MMLUExample]:
    """Convert an entire Hugging Face dataset split into :class:`MMLUExample` objects."""
    return [
        to_mmlu_example(dataset[int(idx)], uid=str(idx))
        for idx in range(len(dataset))
    ]


def load_dev_few_shot_examples(
    subject: str,
    *,
    cache_dir: Optional[str] = None,
    load_config: str = "all",
) -> List[MMLUExample]:
    """Load the five dev exemplars for a specific subject."""
    dataset = load_mmlu(
        split="dev",
        subjects=[subject],
        cache_dir=cache_dir,
        load_config=load_config,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"No dev examples found for subject '{subject}'. Provide few_shot_examples explicitly."
        )
    return dataset_to_examples(dataset)


def _format_choices(
    example: MMLUExample,
    *,
    cue_choice: Optional[int] = None,
    cue_symbol: str = VISUAL_CUE,
) -> str:
    lines: List[str] = []
    for idx, choice in enumerate(example.choices):
        cue = f"{cue_symbol} " if cue_choice is not None and idx == cue_choice else ""
        lines.append(f"{OPTION_LABELS[idx]}. {cue}{choice}")
    return "\n".join(lines)


def build_prompt(
    eval_example: MMLUExample,
    *,
    few_shot_examples: Optional[Sequence[MMLUExample]] = None,
    mark_fewshot_correct: bool = False,
    cue_choice: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Construct a full prompt (instruction + few-shot + evaluation example).

    Parameters
    ----------
    eval_example:
        The evaluation question to append after the few-shot context.
    few_shot_examples:
        Optional explicit few-shot exemplars. When omitted, the function will
        load the five dev examples for ``eval_example.subject``.
    mark_fewshot_correct:
        When True, the correct option in each few-shot example is marked with
        the visual cue.
    cue_choice:
        Optional index of the answer choice in the evaluation example that
        should be marked with the cue.

    Returns
    -------
    dict
        {
            "prompt": <string prompt>,
            "few_shot": [ { "cue_choice": int|None, "answer": str, ...}, ... ],
            "evaluation": {
                "cue_choice": int|None,
                "answer_index": int,
                "answer_label": str,
            },
        }
    """
    SYSTEM_PROMPT = (
        "You are given a question and a set of answer choices. Reason step by step, and answer each question by selecting the single best option (A, B, C, or D). Only respond with the answer index and nothing else."
    )

    if few_shot_examples is None:
        if not eval_example.subject:
            raise ValueError(
                "Evaluation example does not include a subject; provide few_shot_examples explicitly."
            )
        few_shot_examples = load_dev_few_shot_examples(eval_example.subject)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    few_shot_metadata: List[Dict[str, Any]] = []
    for order, example in enumerate(few_shot_examples, start=1):
        cue_choice = example.answer_index if mark_fewshot_correct else None
        stem_lines = [
            f"Example {order}:",
            *(["Subject: " + example.subject] if example.subject else []),
            f"Question: {example.question}",
            _format_choices(example, cue_choice=cue_choice),
        ]
        messages.append(
            {
                "role": "user",
                "content": "\n".join(stem_lines),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Answer: {example.answer_label}",
            }
        )
        few_shot_metadata.append(
            {
                "cue_choice": cue_choice,
                "answer_index": example.answer_index,
                "answer_label": example.answer_label,
                "subject": example.subject,
            }
        )

    eval_lines = [
        f"Question: {eval_example.question}",
        _format_choices(eval_example, cue_choice=cue_choice),
    ]
    messages.append(
        {
            "role": "user",
            "content": "\n".join(eval_lines),
        }
    )

    return {
        "messages": messages,
        "few_shot": few_shot_metadata,
        "evaluation": {
            "cue_choice": cue_choice,
            "answer_index": eval_example.answer_index,
            "answer_label": eval_example.answer_label,
            "subject": eval_example.subject,
        },
    }


__all__ = [
    "MMLUExample",
    "MMLU_STEM_SUBJECTS",
    "OPTION_LABELS",
    "VISUAL_CUE",
    "build_prompt",
    "dataset_to_examples",
    "load_dev_few_shot_examples",
    "load_mmlu",
    "to_mmlu_example",
]

