1. Why does the cluster not allow installing flash attn?
2. vLLM TODO list:
  Plan: Enable vLLM for GRPO Training

  1. Modify src/training.py

  - Add parameters to train_grpo(): use_vllm, vllm_gpu_memory_utilization, vllm_tensor_parallel_size
  - Configure GRPOConfig with vLLM backend when enabled:
  use_vllm=True,
  vllm_device="cuda:0",
  vllm_gpu_memory_utilization=0.8,

  2. Modify scripts/run_pipeline.py

  - Add CLI argument --use_vllm (boolean flag)
  - Pass the flag through train_rl stage to all train_grpo() calls

  3. Update scripts/slurm_train_rl.sh

  - Add vLLM environment variables:
  export VLLM_ATTENTION_BACKEND=FLASHINFER  # or FLASH_ATTN
  - Add --use_vllm flag to the accelerate launch command

  4. Verify dependencies

  - Ensure vLLM is in requirements (already present based on exploration)

  Key Benefits

  - vLLM provides 2-4x faster generation during rollouts
  - Better GPU memory utilization with PagedAttention
  - TRL's GRPOTrainer natively supports vLLM backend

3. Data cleaning: What happens if the public model is too weak, does not follow instructions, and answer is not extractable?

4. Create YAML file for exp configs best practice