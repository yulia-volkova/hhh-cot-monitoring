# Utility helpers shared across the MMLU evaluation workflow.

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

ANSWER_PATTERN = re.compile(r"\b([ABCD])(?:\)|\.|:)?\b")
THINK_CLOSE_PATTERN = re.compile(r"</think>", re.IGNORECASE)


def extract_choice_from_output(text: str) -> Optional[str]:
    """Return the final A/B/C/D choice mentioned in ``text`` (uppercased)."""
    if not text:
        return None
    parts = THINK_CLOSE_PATTERN.split(text)
    text = parts[-1] if parts else text
    last_match = None
    for match in ANSWER_PATTERN.finditer(text):
        last_match = match
    return last_match.group(1) if last_match else None


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append ``record`` as a JSON line to ``path`` (creating directories)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: Iterable[dict], *, fieldnames: Iterable[str]) -> None:
    """Write a CSV file from ``rows`` with the provided ``fieldnames``."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def set_seed(seed: int) -> None:
    """Seed Python, numpy, and torch (if present) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
