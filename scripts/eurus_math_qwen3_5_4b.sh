uv run tinker_cookbook/recipes/math_rl/train.py  \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3.5-4B" \
    env=eurus_math \
    lora_rank=1 \
    max_tokens=4096 \
    group_size=8 \
    groups_per_batch=64 