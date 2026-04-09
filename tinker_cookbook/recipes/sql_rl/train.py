"""Training script for multi-turn text-to-SQL RL.

Usage:
    python -m tinker_cookbook.recipes.sql_rl.train \
        --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
        --train_data_path /path/to/train.parquet \
        --db_root_path /path/to/db_files/data \
        --max_turns 6 \
        --group_size 5 \
        --groups_per_batch 50

The script expects parquet data produced by
``SkyRL/examples/train/synsql/download_data.py``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import chz
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.sql_rl.sql_env import SynSQLDatasetBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config, StreamMinibatchConfig, main
from tinker_cookbook.rl.types import RLDatasetBuilder

import dotenv
dotenv.load_dotenv()


logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for SQL RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Data configuration
    train_data_path: str = "synsql_dataset.parquet"
    val_data_path: str | None = None
    db_root_path: str = ""
    seed: int = 0

    # Environment configuration
    max_turns: int = 6
    max_trajectory_tokens: int = 32 * 1024

    # Training hyperparameters
    group_size: int = 5
    groups_per_batch: int = 50
    learning_rate: float = 1e-6
    max_tokens: int = 3000
    temperature: float = 0.6
    kl_penalty_coef: float = 0.0

    num_substeps: int = 1

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals & checkpointing
    eval_every: int = 20
    save_every: int = 20

    # Service
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None

    stream_minibatch_config: StreamMinibatchConfig | None = None

    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    max_steps: int | None = None


def get_dataset_builder(
    train_data_path: str,
    val_data_path: str | None,
    db_root_path: str,
    model_name: str,
    renderer_name: str,
    batch_size: int,
    group_size: int,
    max_turns: int,
    max_trajectory_tokens: int,
    seed: int,
) -> RLDatasetBuilder:
    return SynSQLDatasetBuilder(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        db_root=db_root_path,
        model_name=model_name,
        renderer_name=renderer_name,
        batch_size=batch_size,
        group_size=group_size,
        max_turns=max_turns,
        max_trajectory_tokens=max_trajectory_tokens,
        seed=seed,
    )


async def cli_main(cli_config: CLIConfig) -> None:
    """Convert CLI config to full Config and run training."""

    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"sql-{model_tag}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr"
        f"-{cli_config.group_size}group-{cli_config.groups_per_batch}batch"
        f"-{cli_config.max_turns}turns-{cli_config.loss_fn}"
        f"-seed{cli_config.seed}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/sql_rl/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            train_data_path=cli_config.train_data_path,
            val_data_path=cli_config.val_data_path,
            db_root_path=cli_config.db_root_path,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            batch_size=cli_config.groups_per_batch,
            group_size=cli_config.group_size,
            max_turns=cli_config.max_turns,
            max_trajectory_tokens=cli_config.max_trajectory_tokens,
            seed=cli_config.seed,
        ),
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=cli_config.groups_per_batch,
            num_minibatches=cli_config.stream_minibatch_config.num_minibatches,
        )
        if cli_config.stream_minibatch_config is not None
        else None,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        max_steps=cli_config.max_steps,
        ttl_seconds=7*24*3600,  # 7 days, effectively no TTL for the duration of the experiment
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
