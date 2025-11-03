import chz
from tinker_cookbook.recipes.sql_rl.sql_env import BIRDDataset
from tinker_cookbook.rl import train
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
import os
import tinker
import asyncio
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-8B"
    max_tokens: int = 3000
    data_path: str | None = None
    db_path: str | None = None
    output_path: str
    timeout: int = 30
    sampler_path: str
    test_data: str = "combined_test.parquet"


async def build_config(cli_config: CLIConfig) -> train.Config:
    if cli_config.output_path is not None:
        os.makedirs(cli_config.output_path, exist_ok=True)
    model_name = cli_config.model_name
    renderer_name = get_recommended_renderer_name(model_name)

    test_dataset = BIRDDataset(
        batch_size=64,
        group_size=1,
        renderer=get_renderer(renderer_name, get_tokenizer(model_name)),
        data_path=f"{cli_config.data_path}/{cli_config.test_data}",
        db_modification_script_path=None,
        timeout=cli_config.timeout,
        db_path=cli_config.db_path,
        split="test",
        n_epochs=1
    )

    print("Creating service client...")
    service_client = tinker.ServiceClient(base_url=None)
    print("Creating training client...")
    training_client = await service_client.create_lora_training_client_async(
        cli_config.model_name, rank=32
    )
    print("Creating sampling client...")
    sampling_client = training_client.create_sampling_client(cli_config.sampler_path)

    return RLTestSetEvaluator(test_dataset, max_tokens=cli_config.max_tokens, dump_path=cli_config.output_path), sampling_client

async def main():
    cli_config = chz.entrypoint(CLIConfig)
    evaluator, sampling_client = await build_config(cli_config)
    
    print("Starting evaluation...")
    metric = await evaluator(sampling_client)
    print(f"Evaluation completed. Metric: {metric}")
   


if __name__ == "__main__":
    asyncio.run(main())