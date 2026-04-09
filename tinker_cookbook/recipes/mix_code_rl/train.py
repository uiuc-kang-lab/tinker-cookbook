"""
Training script for mix-code RL.

Usage:
    SKYRL_PATH=/root/SkyRL python -m tinker_cookbook.recipes.mix_code_rl.train \
        dataset_path=/workspace/data/mix_code/train_data.parquet \
        model_name=Qwen/Qwen3.5-4B \
        max_tokens=4096

See CLIConfig fields for all available options.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import chz
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.mix_code_rl.mix_code_env import MixCodeDatasetBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config, StreamMinibatchConfig, main
from tinker_cookbook.rl.types import RLDatasetBuilder

import dotenv
dotenv.load_dotenv()


logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for mix-code RL training."""

    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Dataset
    dataset_path: str = "/workspace/data/mix_code/train_data.parquet"
    seed: int = 0
    test_fraction: float = 0.0
    timeout: int = 6  # per-test-case timeout in seconds

    # Training
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 4096
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals / checkpoints
    eval_every: int = 20
    save_every: int = 20

    # Service
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Advanced
    max_steps_off_policy: int | None = None
    stream_minibatch_config: StreamMinibatchConfig | None = None
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    max_steps: int | None = None


def get_dataset_builder(
    dataset_path: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
    test_fraction: float = 0.0,
    timeout: int = 6,
) -> RLDatasetBuilder:
    return MixCodeDatasetBuilder(
        dataset_path=dataset_path,
        batch_size=batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        group_size=group_size,
        seed=seed,
        test_fraction=test_fraction,
        timeout=timeout,
    )


async def cli_main(cli_config: CLIConfig):
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )
    model_name = cli_config.model_name.replace("/", "-")
    run_name = (
        f"mix_code-{model_name}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-{cli_config.loss_fn}-"
        f"seed{cli_config.seed}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/mix_code_rl/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            dataset_path=cli_config.dataset_path,
            batch_size=cli_config.groups_per_batch,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
            test_fraction=cli_config.test_fraction,
            timeout=cli_config.timeout,
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
        ttl_seconds=7 * 24 * 3600,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
