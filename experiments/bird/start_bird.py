import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.sql_rl.sql_env import SQLEnv, BIRDDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3"
    group_size: int = 8
    batch_size: int = 64
    learning_rate: float = 1e-6
    max_tokens: int = 3000
    eval_every: int = 5
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None


def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{model_name}-{cli_config.group_size}group-{cli_config.batch_size}batch-{cli_config.learning_rate}lr-{date_and_time}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/data/daniel_kang_group/rl_noise/tinker/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    dataset_builder = BIRDDatasetBuilder(
        batch_size=cli_config.batch_size,
        renderer_name=cli_config.renderer_name,
        train_group_size=cli_config.group_size,
        model_name=cli_config.model_name,
        data_path="/data/daniel_kang_group/rl_noise/data/bird/",
        db_path="/data/daniel_kang_group/rl_noise/data/bird/databases",
        timeout=30
    )

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
