import json
from pathlib import Path

import numpy as np

from utils import append_jsonl, extract_choice_from_output, set_seed, write_csv


def test_extract_choice_from_output_identifies_last_answer_letter():
    assert extract_choice_from_output("The answer is C.") == "C"
    assert extract_choice_from_output("I choose a then switch to D") == "D"
    assert extract_choice_from_output("<think>I might say A</think>Final answer: B") == "B"
    assert extract_choice_from_output("Option D) looks best") == "D"
    assert extract_choice_from_output("A. Maybe B. Final answer: c") == "B"
    assert extract_choice_from_output("<think> ABCD </think>No answer here") is None


def test_append_jsonl_creates_file(tmp_path: Path):
    record = {"question": "q1", "answer": "B"}
    output = tmp_path / "results" / "data.jsonl"
    append_jsonl(output, record)
    data = output.read_text(encoding="utf-8").strip()
    assert json.loads(data) == record


def test_write_csv(tmp_path: Path):
    rows = [{"id": 1, "label": "A"}, {"id": 2, "label": "B"}]
    output = tmp_path / "summary.csv"
    write_csv(output, rows, fieldnames=["id", "label"])
    contents = output.read_text(encoding="utf-8").strip().splitlines()
    assert contents[0] == "id,label"
    assert contents[1] == "1,A"


def test_set_seed_affects_numpy():
    set_seed(123)
    first = np.random.rand()
    set_seed(123)
    second = np.random.rand()
    assert first == second
