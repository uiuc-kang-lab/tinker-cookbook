from tinker_cookbook.rl.train import Config as TrainConfig
from tinker_cookbook.rl.train import main as train
from tinker_cookbook.recipes.math_rl.math_env import ProblemGroupBuilder, MathEnv
from typing import Literal, cast
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers
from tinker_cookbook import model_info
import asyncio
import chz
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from datasets import Dataset, load_dataset
import math
from functools import partial

@chz.chz
class Config:
    data_path: str = "/data/tinker"
    log_path: str = "/log/tinker"
    model_name: str = "Qwen/Qwen3-8B"
    checkpoint_path: str | None = None
    batch_size: int = 64
    group_size: int = 16
    learning_rate: float = 1e-6
    lora_rank: int = 32
    max_tokens: int = 3072
    use_kl: bool = True
    kl_penalty_coef: float = 0.003
    kl_discount_factor: float = 0
    num_substeps: int = 1
    wandb_project: str = "tinker-deepscaler"
    wandb_name: str = "test"
    remove_constant_reward_groups: bool = False
    eval_interval: int = 10
    save_interval: int = 10



class DeepScalerDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        data_path: str,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.ds = cast(Dataset, load_dataset("parquet", data_files=data_path, keep_in_memory=True)["train"])
        if split == "train":
            self.ds = self.ds.shuffle(seed=0)
        if split == "test":
            self.ds = self.ds.select(range(64))
        # select the first 100 data

        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    @classmethod
    def question_suffix(cls) -> str:
        return ""

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x["prompt"][0]["content"]
        answer = x["reward_spec"]["ground_truth"]
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="deepscaler",
        )

@chz.chz
class DeepScalerDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    data_path: str

    async def __call__(self) -> tuple[DeepScalerDataset, DeepScalerDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        train_dataset = DeepScalerDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split="train",
                data_path=f"{self.data_path}/deepscaler_train.parquet"
            )
        test_dataset = DeepScalerDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split="test",
                data_path=f"{self.data_path}/deepscaler_test.parquet"
            )
        return (train_dataset, test_dataset)


def main(config: Config):
    config = TrainConfig(
        learning_rate=1.0e-6,
        dataset_builder=DeepScalerDatasetBuilder(
            batch_size=64, 
            model_name_for_tokenizer=config.model_name, 
            renderer_name=model_info.get_recommended_renderer_name(config.model_name), 
            group_size=config.group_size, 
            convo_prefix=None,
            data_path=config.data_path
        ),
        model_name=config.model_name,
        max_tokens=config.max_tokens,
        compute_post_kl=config.use_kl,
        evaluator_builders=[],
        lora_rank=config.lora_rank,
        kl_penalty_coef=config.kl_penalty_coef,
        kl_discount_factor=config.kl_discount_factor,
        num_substeps=config.num_substeps,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        log_path=config.log_path,
        remove_constant_reward_groups=config.remove_constant_reward_groups,
        eval_every=config.eval_interval,
        save_every=config.save_interval,
        load_checkpoint_path=config.checkpoint_path,
    )

    asyncio.run(train(config))

if __name__ == "__main__":
    chz.nested_entrypoint(main)

