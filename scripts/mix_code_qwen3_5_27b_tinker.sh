uv run tinker_cookbook/recipes/mix_code_rl/train.py  \
    model_name="Qwen/Qwen3.5-27B" \
    lora_rank=64 \
    max_tokens=4096 \
    group_size=8 \
    groups_per_batch=64 \
    wandb_project=rl_bounds_code \
    max_steps=200 \
    save_every=10 \
    log_path=/workspace/tinker/mix_code_qwen3_5_27b_r64 \
    wandb_name=mix_code_qwen3_5_27b_r64 > mix_code_qwen3_5_27b_r64.log 2>&1