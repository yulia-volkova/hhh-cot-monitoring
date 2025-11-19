"""
Training modules for VFT reproduction.
- Supervised Fine-Tuning (SFT) for VFT and BCT
- GRPO for RL reward hacking amplification
"""

import os
import json
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
import wandb

from .vft_dataset import extract_answer


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Format as chat
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions. Think through the problem step by step, then provide your final answer. Format your response as reasoning followed by \"Answer: X\" where X is A, B, C, or D."},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]

        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # Create labels (mask prompt tokens)
        input_ids = encodings["input_ids"]
        labels = input_ids.copy()

        # Find where assistant response starts and mask everything before
        prompt_messages = messages[:-1] + [{"role": "assistant", "content": ""}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        # Mask prompt tokens in labels
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }


def train_sft(
    model_name: str,
    train_data: List[Dict],
    val_data: List[Dict],
    output_dir: str,
    num_epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-5,
    warmup_steps: int = 10,
    gradient_accumulation_steps: int = 8,
    use_lora: bool = False,
    wandb_project: str = "vft-reproduction",
    run_name: str = "sft",
):
    """
    Train model with supervised fine-tuning.

    Args:
        model_name: HuggingFace model name
        train_data: Training examples
        val_data: Validation examples
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Total batch size
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        gradient_accumulation_steps: Gradient accumulation
        use_lora: Whether to use LoRA
        wandb_project: W&B project name
        run_name: W&B run name
    """
    # Initialize wandb
    wandb.init(project=wandb_project, name=run_name)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Create datasets
    train_dataset = SFTDataset(train_data, tokenizer)
    val_dataset = SFTDataset(val_data, tokenizer)

    # Calculate per-device batch size
    num_gpus = torch.cuda.device_count()
    per_device_batch_size = batch_size // (num_gpus * gradient_accumulation_steps)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=10,
        bf16=True,
        report_to="wandb",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final"))

    wandb.finish()


class RLDataset(Dataset):
    """Dataset for RL training."""

    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def reward_function(
    responses: List[str],
    prompts: List[Dict],
) -> List[float]:
    """
    Compute rewards based on the paper's reward function.

    R(M(x)|x) = 1[M(x) = y_b] if x in B (cued prompts)
                1[M(x) = y_true] if x in U (uncued prompts)

    Args:
        responses: Model responses
        prompts: Prompt metadata including cued_answer and correct_answer

    Returns:
        List of rewards
    """
    rewards = []

    for response, prompt_info in zip(responses, prompts):
        model_answer = extract_answer(response)

        if prompt_info["is_cued"]:
            # Reward cue-aligned wrong answer
            if model_answer == prompt_info["cued_answer"]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            # Reward correct answer
            if model_answer == prompt_info["correct_answer"]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

    return rewards


def train_grpo(
    model_path: str,
    train_data: List[Dict],
    output_dir: str,
    num_steps: int = 150,
    batch_size: int = 256,
    num_rollouts: int = 8,
    learning_rate: float = 1e-5,
    warmup_steps: int = 10,
    kl_coef: float = 0.001,
    gradient_accumulation_steps: int = 8,
    wandb_project: str = "vft-reproduction",
    run_name: str = "grpo",
):
    """
    Train model with GRPO for reward hacking amplification.

    Args:
        model_path: Path to pre-trained model (Baseline, VFT, or BCT)
        train_data: Training examples with cues
        output_dir: Where to save checkpoints
        num_steps: Number of training steps
        batch_size: Batch size (number of prompts)
        num_rollouts: Number of rollouts per prompt
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        kl_coef: KL penalty coefficient
        gradient_accumulation_steps: Gradient accumulation
        wandb_project: W&B project name
        run_name: W&B run name
    """
    # Initialize wandb
    wandb.init(project=wandb_project, name=run_name)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Calculate per-device batch size
    num_gpus = torch.cuda.device_count()
    per_device_batch_size = batch_size // (num_gpus * gradient_accumulation_steps)

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=num_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=10,
        bf16=True,
        report_to="wandb",
        num_generations=num_rollouts,
        max_completion_length=1024,
        beta=kl_coef,  # KL penalty
    )

    # Format data for GRPO
    def format_prompt(example):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions. Think through the problem step by step, then provide your final answer. Format your response as reasoning followed by \"Answer: X\" where X is A, B, C, or D."},
            {"role": "user", "content": example["prompt"]},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Create dataset
    formatted_data = []
    for example in train_data:
        formatted_data.append({
            "prompt": format_prompt(example),
            "is_cued": example["is_cued"],
            "cued_answer": example.get("cued_answer"),
            "correct_answer": example["correct_answer"],
        })

    # Custom reward function wrapper for GRPO
    def compute_rewards(completions, prompts):
        prompt_infos = [formatted_data[i % len(formatted_data)] for i in range(len(completions))]
        return reward_function(completions, prompt_infos)

    # GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=RLDataset(formatted_data),
        processing_class=tokenizer,
        reward_funcs=compute_rewards,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final"))

    wandb.finish()
