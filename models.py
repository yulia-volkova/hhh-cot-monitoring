"""Lightweight wrappers for interacting with chat-based reasoning models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from utils import extract_choice_from_output

try:  # Optional dependencies are loaded lazily.
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is optional at runtime.
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = AutoTokenizer = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False

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


def _ensure_transformers() -> None:
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers and torch must be installed to use ReasoningTransformersClient"
        )


def _build_chat_prompt(tokenizer, messages: Sequence[Dict[str, str]]) -> str:
    """Use the tokenizer's chat template to format a conversation."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


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
        _ensure_transformers()

        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[operator]
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(  # type: ignore[operator]
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

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[union-attr]
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
        prompt = _build_chat_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)  # type: ignore[operator]

        with torch.no_grad():  # type: ignore[attr-defined]
            output = self.model.generate(  # type: ignore[call-arg]
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        generated_tokens = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        if stop:
            for token in stop:
                if token in text:
                    text = text.split(token, 1)[0].strip()
                    break

        choice = extract_choice_from_output(text)
        reasoning = None
        if "</think>" in text.lower():
            # Split on the closing tag, preserving case.
            parts = text.split("</think>", 1)
            reasoning = parts[0]
        return GenerationResult(
            text=text,
            choice=choice,
            reasoning=reasoning,
            raw_output=output,
            prompt=prompt,
        )


class ReasoningVLLMClient:
    """Run chat reasoning models via vLLM."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        *,
        tokenizer_name: Optional[str] = None,
        trust_remote_code: bool = True,
        **llm_kwargs: Any,
    ) -> None:
        try:  # pragma: no cover - vLLM optional dependency
            from vllm import LLM  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("vllm must be installed to use ReasoningVLLMClient") from exc

        _ensure_transformers()

        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[operator]
            tokenizer_name or model_name,
            trust_remote_code=trust_remote_code,
        )
        self.llm = LLM(model=model_name, trust_remote_code=trust_remote_code, **llm_kwargs)

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[Sequence[str]] = None,
    ) -> GenerationResult:
        try:  # pragma: no cover
            from vllm import SamplingParams  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("vllm must be installed to use ReasoningVLLMClient") from exc

        prompt = _build_chat_prompt(self.tokenizer, messages)
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        outputs = self.llm.generate([prompt], sampling_params=sampling_params)
        generated = outputs[0].outputs[0].text.strip()

        choice = extract_choice_from_output(generated)
        reasoning = None
        if "</think>" in generated.lower():
            parts = generated.split("</think>", 1)
            reasoning = parts[0]
        return GenerationResult(
            text=generated,
            choice=choice,
            reasoning=reasoning,
            raw_output=outputs,
            prompt=prompt,
        )
