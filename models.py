"""Lightweight wrappers for interacting with chat-based reasoning models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from utils import extract_choice_from_output

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = [
    "GenerationResult",
    "ReasoningTransformersClient",
    "ReasoningVLLMClient",
]


@dataclass
class GenerationResult:
    """Container for a single generation call."""

    text: str
    choice: Optional[str]
    reasoning: Optional[str]
    raw_output: Any
    prompt: str


def _build_chat_prompt(tokenizer, messages: Sequence[Dict[str, str]]) -> str:
    """Use the tokenizer's chat template to format a conversation."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _apply_stop_sequences(text: str, stop: Optional[Sequence[str]]) -> str:
    if not stop:
        return text
    for token in stop:
        if token and token in text:
            return text.split(token, 1)[0].strip()
    return text


def _extract_reasoning(text: str) -> Optional[str]:
    lowered = text.lower()
    if "</think>" in lowered:
        parts = text.split("</think>", 1)
        return parts[0]
    return None


class ReasoningTransformersClient:
    """Run chat reasoning models via Hugging Face transformers."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        *,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "auto",
        trust_remote_code: bool = True,
        **model_kwargs: Any,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if hasattr(self.model, "config"):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Sequence[str]] = None,
    ) -> GenerationResult:
        return self.generate_batch(
            [messages],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )[0]

    def generate_batch(
        self,
        batch_messages: Sequence[Sequence[Dict[str, str]]],
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Sequence[str]] = None,
    ) -> List[GenerationResult]:
        prompts = [_build_chat_prompt(self.tokenizer, messages) for messages in batch_messages]
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in tokenized.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        batch_results: List[GenerationResult] = []
        for prompt_str, generated_ids, prompt_len in zip(
            prompts,
            output,
            input_lengths,
        ):
            new_tokens = generated_ids[prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            text = _apply_stop_sequences(text, stop)
            choice = extract_choice_from_output(text)
            reasoning = _extract_reasoning(text)
            batch_results.append(
                GenerationResult(
                    text=text,
                    choice=choice,
                    reasoning=reasoning,
                    raw_output=generated_ids.detach().cpu(),
                    prompt=prompt_str,
                )
            )
        return batch_results


class ReasoningVLLMClient:
    """Run chat reasoning models via vLLM."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        *,
        tokenizer_name: Optional[str] = None,
        trust_remote_code: bool = True,
        seed: Optional[int] = None,
        **llm_kwargs: Any,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("vllm must be installed to use ReasoningVLLMClient") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name,
            trust_remote_code=trust_remote_code,
        )
        self.llm = LLM(model=model_name, trust_remote_code=trust_remote_code, **llm_kwargs)
        self.seed = seed

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Sequence[str]] = None,
    ) -> GenerationResult:
        return self.generate_batch(
            [messages],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )[0]

    def generate_batch(
        self,
        batch_messages: Sequence[Sequence[Dict[str, str]]],
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Sequence[str]] = None,
    ) -> List[GenerationResult]:
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError("vllm must be installed to use ReasoningVLLMClient") from exc

        prompts = [_build_chat_prompt(self.tokenizer, messages) for messages in batch_messages]
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            seed=self.seed,
        )
        outputs = self.llm.generate(prompts, sampling_params=sampling_params)

        batch_results: List[GenerationResult] = []
        for prompt_str, output in zip(prompts, outputs):
            if not output.outputs:
                text = ""
            else:
                text = output.outputs[0].text.strip()
            text = _apply_stop_sequences(text, stop)
            choice = extract_choice_from_output(text)
            reasoning = _extract_reasoning(text)
            batch_results.append(
                GenerationResult(
                    text=text,
                    choice=choice,
                    reasoning=reasoning,
                    raw_output=output,
                    prompt=prompt_str,
                )
            )
        return batch_results
