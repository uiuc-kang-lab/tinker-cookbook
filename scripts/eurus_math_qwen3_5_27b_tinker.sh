uv run tinker_cookbook/recipes/math_rl/train.py  \
    model_name="Qwen/Qwen3.5-27B" \
    env=eurus_math \
    lora_rank=32 \
    max_tokens=4096 \
    group_size=8 \
    num_epochs=1 \
    groups_per_batch=64 \
    wandb_project=rl_bounds \
    wandb_name=eurus_math_qwen3_5_27b_r32 > eurus_math_qwen3_5_27b_r32.log 2>&1