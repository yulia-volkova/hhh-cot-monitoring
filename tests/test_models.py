from typing import Dict, List

import pytest

from models import GenerationResult, ReasoningTransformersClient, _build_chat_prompt


class DummyTokenizer:
    def apply_chat_template(self, messages: List[Dict[str, str]], *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert not tokenize
        assert add_generation_prompt
        parts = []
        for msg in messages:
            parts.append(f"[{msg['role']}] {msg['content']}")
        parts.append("[assistant]")
        return "\n".join(parts)


def test_generation_result_dataclass():
    result = GenerationResult(text="hello", choice="A", reasoning="<think>hi</think>", raw_output=None, prompt="prompt")
    assert result.text == "hello"
    assert result.choice == "A"
    assert result.reasoning == "<think>hi</think>"
    assert result.prompt == "prompt"


def test_build_chat_prompt_with_dummy_tokenizer():
    tokenizer = DummyTokenizer()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    prompt = _build_chat_prompt(tokenizer, messages)
    assert prompt.startswith("[system] You are helpful.")
    assert prompt.endswith("[assistant]")


@pytest.mark.slow
def test_transformers_client_end_to_end():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    client = ReasoningTransformersClient(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    messages = [
        {"role": "system", "content": "Answer math questions."},
        {
            "role": "user",
            "content": "Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6",
        },
    ]
    result = client.generate(messages, max_new_tokens=512, temperature=0.0)
    print(result)
    assert isinstance(result.text, str)
    assert isinstance(result.prompt, str)
    assert result.reasoning is None or isinstance(result.reasoning, str)
    assert result.choice in {None, "A", "B", "C", "D"}
