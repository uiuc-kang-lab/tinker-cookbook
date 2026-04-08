"""SQL environment, dataset, and builders for RL training.

Provides:
- ``SQLEnvGroupBuilder``: builds a group of multi-turn SQL environments for
  a single text-to-SQL problem.
- ``SynSQLDataset`` / ``SynSQLDatasetBuilder``: loads parquet data produced
  by ``download_data.py`` and yields batches of ``SQLEnvGroupBuilder``.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence

import chz
from datasets import load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.recipes.sql_rl.sql_reward import make_sql_reward_fn
from tinker_cookbook.recipes.sql_rl.sql_tool import SQLExecutorTools
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use import build_agent_tool_env

logger = logging.getLogger(__name__)

DB_SUBPATH: dict[str, str] = {
    "synsql": "SynSQL-2.5M/databases",
    "spider": "spider/database",
    "bird": "bird/train/train_databases",
}


def get_db_file(db_root: str, task: str, db_id: str) -> str:
    """Resolve the SQLite database file path from root, task type, and db_id."""
    subpath = DB_SUBPATH.get(task)
    if subpath is None:
        raise ValueError(
            f"Unknown task type: {task!r}. Expected one of {list(DB_SUBPATH.keys())}"
        )
    return os.path.join(db_root, subpath, db_id, f"{db_id}.sqlite")


# ---------------------------------------------------------------------------
# EnvGroupBuilder
# ---------------------------------------------------------------------------


class SQLEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of multi-turn SQL environments for one problem.

    Each call to ``make_envs`` creates ``group_size`` independent environments
    that share the same problem but have separate tool instances and state.
    """

    def __init__(
        self,
        *,
        prompt_messages: list[Message],
        gold_sql: str,
        db_file: str,
        model_name: str,
        renderer_name: str,
        group_size: int,
        max_turns: int = 6,
        max_trajectory_tokens: int = 32 * 1024,
    ):
        self.prompt_messages = prompt_messages
        self.gold_sql = gold_sql
        self.db_file = db_file
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        envs: list[Env] = []
        for _ in range(self.group_size):
            tool_instance = SQLExecutorTools(
                db_file=self.db_file, max_turns=self.max_turns
            )
            reward_fn = make_sql_reward_fn(self.gold_sql, self.db_file)

            # Use the prompt messages from the dataset directly. The system
            # message already contains tool specifications in the format
            # expected by the model (e.g. Qwen3.5 XML tool-call format).
            initial_messages = list(self.prompt_messages)

            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=[tool_instance.execute_sql],
                    initial_messages=initial_messages,
                    reward_fn=reward_fn,
                    max_turns=self.max_turns,
                    failed_parse_reward=-1.0,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                )
            )
        return envs

    def logging_tags(self) -> list[str]:
        return ["sql"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SynSQLDataset(RLDataset):
    """Dataset that loads parquet data produced by ``download_data.py``.

    Expected parquet columns (subset used here):
    - ``prompt``: list of message dicts (system + user)
    - ``reward_spec``: dict with ``ground_truth`` key
    - ``db_id``: database identifier
    - ``data``: task type (``"synsql"``, ``"spider"``, ``"bird"``)
    """

    def __init__(
        self,
        *,
        data_path: str,
        db_root: str,
        model_name: str,
        renderer_name: str,
        batch_size: int,
        group_size: int,
        max_turns: int = 6,
        max_trajectory_tokens: int = 32 * 1024,
        seed: int = 0,
    ):
        ds = load_dataset("parquet", data_files=data_path, keep_in_memory=True)[
            "train"
        ]
        self.ds = ds.shuffle(seed=seed)
        self.db_root = db_root
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Batch index out of range"

        builders: list[EnvGroupBuilder] = []
        for row in self.ds.select(range(batch_start, batch_end)):
            builder = self._make_builder(row)
            if builder is not None:
                builders.append(builder)
        return builders

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_builder(self, row: dict) -> SQLEnvGroupBuilder | None:
        try:
            prompt_messages: list[Message] = row["prompt"]
            gold_sql: str = row["reward_spec"]["ground_truth"]
            db_id: str = row["db_id"]
            task: str = row["data"]
            db_file = get_db_file(self.db_root, task, db_id)
        except (KeyError, TypeError) as e:
            logger.warning("Skipping malformed row: %s", e)
            return None

        return SQLEnvGroupBuilder(
            prompt_messages=prompt_messages,
            gold_sql=gold_sql,
            db_file=db_file,
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            group_size=self.group_size,
            max_turns=self.max_turns,
            max_trajectory_tokens=self.max_trajectory_tokens,
        )


# ---------------------------------------------------------------------------
# DatasetBuilder (chz-compatible for CLI)
# ---------------------------------------------------------------------------


@chz.chz
class SynSQLDatasetBuilder(RLDatasetBuilder):
    """Builds train (and optionally validation) SQL datasets from parquet files."""

    train_data_path: str
    val_data_path: str | None = None
    db_root: str = ""
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    renderer_name: str = "qwen3_5"
    batch_size: int = 50
    group_size: int = 5
    max_turns: int = 6
    max_trajectory_tokens: int = 32 * 1024
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train_ds = SynSQLDataset(
            data_path=self.train_data_path,
            db_root=self.db_root,
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            batch_size=self.batch_size,
            group_size=self.group_size,
            max_turns=self.max_turns,
            max_trajectory_tokens=self.max_trajectory_tokens,
            seed=self.seed,
        )
        val_ds: RLDataset | None = None
        if self.val_data_path is not None:
            val_ds = SynSQLDataset(
                data_path=self.val_data_path,
                db_root=self.db_root,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                batch_size=self.batch_size,
                group_size=1,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                seed=self.seed,
            )
        return train_ds, val_ds
