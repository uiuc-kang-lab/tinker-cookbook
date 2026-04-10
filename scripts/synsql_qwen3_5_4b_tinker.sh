uv run tinker_cookbook/recipes/sql_rl/train.py  \
    model_name="Qwen/Qwen3.5-4B" \
    train_data_path=/workspace/data/sql/train.parquet \
    db_root_path=/workspace/data/sql/databases \
    max_turns=6 \
    lora_rank=64 \
    learning_rate=1e-4 \
    max_tokens=4096 \
    group_size=8 \
    groups_per_batch=64 \
    wandb_project=rl_bounds_synsql \
    log_path=/workspace/tinker/synsql_qwen3_5_4b_r64_lr1e-4 \
    max_steps=100 \
    wandb_name=synsql_qwen3_5_4b_r64_lr1e-4 > synsql_qwen3_5_4b_r64_lr1e-4.log 2>&1