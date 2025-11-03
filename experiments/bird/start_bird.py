import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.sql_rl.sql_env import SQLEnv, BIRDDatasetBuilder
from tinker_cookbook.rl import train
from tinker_cookbook.model_info import get_recommended_renderer_name


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-8B"
    group_size: int = 8
    batch_size: int = 64
    learning_rate: float = 5e-5
    max_output_tokens_per_turn: int = 3000
    max_input_tokens: int = 32768
    eval_every: int = 10
    save_every: int = 20
    wandb_project: str | None = "tinker-sql"
    wandb_name: str | None = None
    log_path: str | None = None
    data_path: str | None = None
    db_path: str | None = None
    db_modification_script_path: str | None = None
    checkpoint_path: str | None = None
    add_noise: str | None = None
    timeout: int = 30
    n_epochs: int = 1
    num_data: int = -1
    use_convo_prefix: bool = True
    use_system_prompt: bool = True
    renderer_name: str = "default"


def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{model_name}-{cli_config.group_size}group-{cli_config.batch_size}batch-{cli_config.learning_rate}lr-{date_and_time}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/mydata/tinker/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    dataset_builder = BIRDDatasetBuilder(
        batch_size=cli_config.batch_size,
        renderer_name=get_recommended_renderer_name(model_name) if cli_config.renderer_name == "default" else cli_config.renderer_name,
        train_group_size=cli_config.group_size,
        model_name=cli_config.model_name,
        data_path=cli_config.data_path,
        db_path=cli_config.db_path,
        timeout=cli_config.timeout,
        add_noise=cli_config.add_noise,
        n_epochs=cli_config.n_epochs,
        db_modification_script_path=cli_config.db_modification_script_path,
        num_data=cli_config.num_data,
        use_convo_prefix=cli_config.use_convo_prefix,
        use_system_prompt=cli_config.use_system_prompt,
        max_output_tokens_per_turn=cli_config.max_output_tokens_per_turn,
    )

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_output_tokens_per_turn,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        n_epochs=1,
        save_every=cli_config.save_every,
        load_checkpoint_path=cli_config.checkpoint_path,
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
