"""
Mix-general RL environment and dataset for tinker-cookbook.

Loads a parquet dataset produced by SkyRL's download_datasets.py and creates
ProblemEnv instances that use the same grading logic as SkyRL's GeneralEnv.
"""

import math
from collections.abc import Sequence
from functools import partial
from typing import Literal

import chz
from datasets import Dataset, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tinker_cookbook.recipes.mix_general_rl.grading import THOUGHT_DELIMITER_END, grade
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MixGeneralEnv(ProblemEnv):
    """Single-problem environment for mix-general tasks (MMLU, WebInstruct,
    LegalBench, MedQA, CEVAL, ARC, LogiQA).

    The prompt already contains a system message instructing the model to use
    \\boxed{}, so ``no_question_suffix=True`` by default.
    """

    def __init__(
        self,
        problem: str,
        answer: str,
        method: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        timeout: float = 10.0,
    ):
        super().__init__(renderer, convo_prefix)
        self.problem = problem
        self.answer = answer
        self.method = method
        self.timeout = timeout

    def get_question(self) -> str:
        return self.problem

    def check_format(self, sample_str: str) -> bool:
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        # Strip <think>...</think> reasoning before grading
        if THOUGHT_DELIMITER_END in sample_str:
            sample_str = sample_str.split(THOUGHT_DELIMITER_END, 1)[1]
        try:
            model_answer = extract_boxed(sample_str)
        except ValueError:
            return False

        # Process ground truth (may contain \boxed{})
        ground_truth = self.answer
        if "\\boxed" in ground_truth:
            try:
                ground_truth = extract_boxed(ground_truth)
            except ValueError:
                pass

        return grade(ground_truth, model_answer, self.method, timeout=self.timeout)

    def get_reference_answer(self) -> str:
        return self.answer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MixGeneralDataset(RLDataset):
    """Wraps a parquet dataset produced by download_datasets.py."""

    def __init__(
        self,
        ds: Dataset,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        timeout: float = 10.0,
    ):
        self.ds = ds
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.timeout = timeout

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, row: dict, group_size: int
    ) -> ProblemGroupBuilder | None:
        # The prompt field is a list of message dicts (system + user).
        # We split into convo_prefix (system message) and the user question.
        prompt_messages: list[dict] = row["prompt"]
        convo_prefix: list[renderers.Message] = []
        question = ""
        for msg in prompt_messages:
            if msg["role"] == "user":
                question = msg["content"]
            else:
                convo_prefix.append(msg)

        if not question:
            logger.warning(f"No user message found in prompt: {prompt_messages}")
            return None

        reward_spec: dict = row["reward_spec"]
        method = reward_spec["method"]
        ground_truth = reward_spec["ground_truth"]
        data_source = row.get("data_source", "unknown")

        return ProblemGroupBuilder(
            env_thunk=partial(
                MixGeneralEnv,
                problem=question,
                answer=str(ground_truth),
                method=method,
                renderer=self.renderer,
                convo_prefix=convo_prefix if convo_prefix else None,
                timeout=self.timeout,
            ),
            num_envs=group_size,
            dataset_name=data_source,
        )


# ---------------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------------


@chz.chz
class MixGeneralDatasetBuilder(RLDatasetBuilder):
    """Builds train (and optionally test) datasets from a parquet file."""

    dataset_path: str  # path to parquet file or HF dataset identifier
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    test_fraction: float = 0.0  # fraction of data to hold out for eval
    timeout: float = 10.0

    async def __call__(self) -> tuple[MixGeneralDataset, MixGeneralDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Load from parquet or HF hub
        if self.dataset_path.endswith(".parquet"):
            ds = Dataset.from_parquet(self.dataset_path)
        else:
            ds = load_dataset(self.dataset_path, split="train")

        ds = ds.shuffle(seed=self.seed)

        test_ds = None
        if self.test_fraction > 0:
            split = ds.train_test_split(test_size=self.test_fraction, seed=self.seed)
            ds = split["train"]
            test_ds = split["test"]

        train_dataset = MixGeneralDataset(
            ds=ds,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            timeout=self.timeout,
        )

        test_dataset = None
        if test_ds is not None:
            test_dataset = MixGeneralDataset(
                ds=test_ds,
                batch_size=self.batch_size,
                group_size=1,
                renderer=renderer,
                timeout=self.timeout,
            )

        return train_dataset, test_dataset
