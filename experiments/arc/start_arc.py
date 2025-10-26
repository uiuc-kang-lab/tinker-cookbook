from tinker_cookbook.rl.train import Config as TrainConfig
from tinker_cookbook.rl.train import main as train
from tinker_cookbook.recipes.math_rl.math_env import ProblemGroupBuilder, MathEnv
from typing import Literal, cast
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers
from tinker_cookbook import model_info
from tinker_cookbook.rl.problem_env import ProblemEnv
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    StepResult,
    RLDataset, 
    RLDatasetBuilder
)
import tinker
import asyncio
import chz
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
    learning_rate: float = 5e-4
    lora_rank: int = 32
    max_tokens: int = 3072
    use_kl: bool = False
    kl_penalty_coef: float = 0
    kl_discount_factor: float = 0
    num_substeps: int = 1
    wandb_project: str = "tinker-arc-bs1"
    wandb_name: str = "test"
    remove_constant_reward_groups: bool = False
    eval_interval: int = 10
    save_interval: int = 10
    noise_rate: float = 0
    n_epochs: int = 1

class ArcEnv(ProblemEnv):
    def __init__(
        self,
        question: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None
    ):
        self.question = question
        self.answer = answer
        self.renderer = renderer
        self.convo_prefix = convo_prefix if convo_prefix is not None else []

    def get_question(self) -> str:
        return self.question

    def check_format(self, sample_str: str) -> bool:
        return True
    
    def check_answer(self, sample_str):
        if "####" not in sample_str:
            # print("Env: Action does not contain ####, returning 0 reward")
            return False
        pred_answer = sample_str.split("####")[-1].split(".")[0].strip().lower()
        # if pred_answer has more than one alphabetic character, return 0
        if sum(c.isalpha() for c in pred_answer) > 1:
            # print("Env: Action contains more than one alphabetic character, returning 0 reward")
            return False
        # extract the first alphabetic character
        pred_answer = "".join([c for c in pred_answer if c.isalpha()])
        if len(pred_answer) == 0:
            # print("Env: Action does not contain any alphabetic characters, returning 0 reward")
            return False
        pred_answer = pred_answer[0]
        if pred_answer == self.answer.lower():
            # print("Env: Action is correct, returning 1 reward")
            return True
        else:
            # print("Env: Action is incorrect, returning 0 reward")
            return False

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        correct_answer = float(self.check_answer(message["content"]))
        return StepResult(
            reward=correct_answer,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "correct": correct_answer,
            },
        )

class ArcDataset(RLDataset):
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
        dataset_name=x["data_source"]
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                ArcEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name=dataset_name,
        )

@chz.chz
class ArcDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    data_path: str
    noise_rate: float

    async def __call__(self) -> tuple[ArcDataset, ArcDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        train_dataset_name = "train.parquet" if self.noise_rate == 0 else f"train_noise_{self.noise_rate:.1f}.parquet"
        print(f"Loading train dataset from {self.data_path}/{train_dataset_name}")
        test_dataset_name = "val.parquet"
        train_dataset = ArcDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split="train",
                data_path=f"{self.data_path}/{train_dataset_name}"
            )
        print(train_dataset.ds[1])
        test_dataset = ArcDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split="test",
                data_path=f"{self.data_path}/{test_dataset_name}"
            )
        return (train_dataset, test_dataset)


def main(config: Config):
    config = TrainConfig(
        learning_rate=config.learning_rate,
        dataset_builder=ArcDatasetBuilder(
            batch_size=64, 
            model_name_for_tokenizer=config.model_name, 
            renderer_name=model_info.get_recommended_renderer_name(config.model_name), 
            group_size=config.group_size, 
            convo_prefix=None,
            data_path=config.data_path,
            noise_rate=config.noise_rate
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
        n_epochs=config.n_epochs,
    )

    asyncio.run(train(config))

if __name__ == "__main__":
    chz.nested_entrypoint(main)

