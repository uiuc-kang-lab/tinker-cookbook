"""
Mix-code RL environment and dataset for tinker-cookbook.

Loads a parquet dataset produced by SkyRL's
``examples/train/mix_code/download_data.py`` and creates ProblemEnv instances
that use the same subprocess-isolated evaluation as SkyRL's MixCodeEnv.

Two evaluation methods are supported:
- "inputs":     stdin/stdout testing with input/output pairs (Eurus format)
- "assertions": assertion-based testing with test code strings (KodCode format)
"""

import asyncio
import math
from collections.abc import Sequence
from functools import partial

import chz
import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.recipes.mix_code_rl.code_grading import compute_score, extract_code_from_model
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder, Action, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MixCodeEnv(ProblemEnv):
    """Single-problem environment for code generation tasks.

    Uses SkyRL's ``compute_score`` for evaluation, which runs model-generated
    code in an isolated subprocess with timeouts and security guards.
    """

    def __init__(
        self,
        problem: str,
        ground_truth: str,
        method: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        timeout: int = 6,
    ):
        # format_coef=0.0: no penalty for missing code blocks — the reward
        # is entirely determined by whether the code passes the tests.
        super().__init__(renderer, convo_prefix, format_coef=0.0)
        self.problem = problem
        self.ground_truth = ground_truth
        self.method = method
        self.timeout = timeout

    def get_question(self) -> str:
        return self.problem

    def check_format(self, sample_str: str) -> bool:
        return extract_code_from_model(sample_str) is not None

    def check_answer(self, sample_str: str) -> bool:
        _, reward, is_timeout = compute_score(
            sample_str, self.ground_truth, self.method, timeout=self.timeout,
        )
        return reward > 0

    def get_reference_answer(self) -> str:
        return f"[{self.method}] {self.ground_truth[:100]}..."

    async def step(self, action: Action) -> StepResult:
        """Override ProblemEnv.step to run check_answer in a thread.

        compute_score() spawns subprocesses and calls p.join() which blocks.
        Without asyncio.to_thread, this freezes the event loop and prevents
        all other rollouts in the same group from making progress.
        """
        convo = self.convo_prefix + [{"role": "user", "content": self.get_question()}]
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        correct_format = float(parse_success) and float(self.check_format(content))
        correct_answer, is_timeout = await asyncio.to_thread(self.check_answer, content)
        correct_answer = float(correct_answer)
        is_timeout = float(is_timeout)
        total_reward = self.format_coef * (correct_format - 1) + correct_answer

        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=convo))
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "reference_answer": self.get_reference_answer(),
                    "format_valid": bool(correct_format),
                    "correct": bool(correct_answer),
                    "format_coef": self.format_coef,
                    "reward": f"{total_reward:.3f}",
                    "timeout": bool(is_timeout),
                },
                caption="Reward components",
            )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
                "timeout": is_timeout,
            },
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MixCodeDataset(RLDataset):
    """Wraps a parquet dataset produced by download_data.py."""

    def __init__(
        self,
        ds: Dataset,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        timeout: int = 6,
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
                MixCodeEnv,
                problem=question,
                ground_truth=str(ground_truth),
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
class MixCodeDatasetBuilder(RLDatasetBuilder):
    """Builds train (and optionally test) datasets from a parquet file."""

    dataset_path: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    test_fraction: float = 0.0
    timeout: int = 6

    async def __call__(self) -> tuple[MixCodeDataset, MixCodeDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        if self.dataset_path.endswith(".parquet"):
            import pyarrow.parquet as pq

            # PyArrow 23 crashes (segfault or OOM) reading nested columns
            # (list<struct>, struct) as a full table.  iter_batches() works
            # on small chunks.  Stream rows through Dataset.from_generator()
            # so only one batch lives in memory at a time.
            path = self.dataset_path

            def _parquet_row_iter():
                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(batch_size=10_000):
                    yield from batch.to_pylist()

            ds = Dataset.from_generator(_parquet_row_iter)
        else:
            ds = load_dataset(self.dataset_path, split="train")

        ds = ds.shuffle(seed=self.seed)

        test_ds = None
        if self.test_fraction > 0:
            split = ds.train_test_split(test_size=self.test_fraction, seed=self.seed)
            ds = split["train"]
            test_ds = split["test"]

        train_dataset = MixCodeDataset(
            ds=ds,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            timeout=self.timeout,
        )

        test_dataset = None
        if test_ds is not None:
            test_dataset = MixCodeDataset(
                ds=test_ds,
                batch_size=self.batch_size,
                group_size=1,
                renderer=renderer,
                timeout=self.timeout,
            )

        return train_dataset, test_dataset
