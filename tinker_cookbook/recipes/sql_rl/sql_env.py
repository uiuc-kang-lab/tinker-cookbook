import random
import pandas as pd
from dataclasses import dataclass
from typing import Sequence

import chz
from tinker import ModelInput
from tinker_cookbook.completers import (
    StopCondition,
)
from tinker_cookbook.renderers import Message, Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
    Metrics
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.sql_rl.sql_utils import verify_format_and_extract, execute_sql_wrapper_single
from tinker_cookbook.recipes.sql_rl.grader import grade
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.recipes.sql_rl.prompts import system_prompt, instruction_prompt, task_overview_prompt
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Literal, cast, Tuple, Any
from functools import partial
import math
import re
import os
import logging

logger = logging.getLogger(__name__)

MAX_TURNS = 5

convo_prefix = [
    Message(role="system", content=system_prompt),
    Message(role="user", content="{db_details}: CREATE TABLE animals (\n`id` integer, -- ID of the animal.\n`species` text, -- species of the animal.\n`age` integer, -- age of the animal.\n`name` text, -- name of the animal.\nprimary key (id)\n);\n\n{external_knowledge}: pig is the species of.\n{question}: how many pigs are in the farm?"),
    Message(role="assistant", content="I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.\n<sql>SELECT COUNT(*) FROM animals WHERE species = 'pig';</sql>"),
    Message(role="user", content="+----------+\n| COUNT(*) |\n+----------+\n|   12     |\n+----------+\n"),
    Message(role="assistant", content="The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.\n<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>")
]

class SQLEnv(Env):
    def __init__(self, question_id: str, question: str, gold_answer: int, grading_method: str, renderer: Renderer, db_file, timeout, db_modification_script: str | None, dump_path: str | None = None, use_convo_prefix: bool = True, use_system_prompt: bool = True, max_output_tokens_per_turn: int = 3072,
    max_input_tokens: int = 32768):
        self.renderer: Renderer = renderer
        self.turns: list[Message] = []
        self.gold_answer: int = gold_answer
        self.db_file = db_file
        self.timeout = timeout
        self.num_turn = 0
        self.question = question
        self.question_id = question_id
        self.db_modification_script = db_modification_script
        self.dump_path = dump_path
        self.grading_method = grading_method
        self.use_convo_prefix = use_convo_prefix
        self.use_system_prompt = use_system_prompt
        self.max_output_tokens_per_turn = max_output_tokens_per_turn
        self.max_input_tokens = max_input_tokens

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    @property
    def _obs(self) -> ModelInput:
        """Get the observation for the player in tokenized form"""
        if self.use_convo_prefix and self.use_system_prompt:
            convo = convo_prefix + [Message(role="assistant", content=self.question)] + self.turns
        elif not self.use_convo_prefix and self.use_system_prompt:
            convo = [Message(role="system", content=system_prompt), Message(role="assistant", content=self.question)] + self.turns
        elif not self.use_system_prompt:
            user_prompt = task_overview_prompt + instruction_prompt + self.question
            convo = [Message(role="user", content=user_prompt)] + self.turns
        # print(f"========== DEBUG ==========\n Current conversation turns: {[{'role': m["role"], 'content': m["content"]} for m in convo]}")
        return self.renderer.build_generation_prompt(convo)

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return self._obs, self.stop_condition

    def _parse_action(self, action: str) -> Tuple[str, str, Any]:
        """
        Parse action string to return tool name and corresponding arguments.

        Expected: <sql>...</sql>
        """
        matches = re.findall(r"<sql>(.*?)</sql>", action, re.DOTALL)
        tool_input = matches[-1] if matches else None
        return tool_input

    def _get_user_turn(self, action_text: str) -> tuple[Message, float, str]:

        # check if there is a sql tool call
        if action_text.endswith('</sql>'):
            # this means this turn is an intermediate step
            sql = self._parse_action(action_text)
            # print(f"Executing SQL: {sql}")
            sql_output = execute_sql_wrapper_single(self.db_file, sql, self.timeout, action_text, self.db_modification_script)
            _, _, pred_results, error, _ = sql_output
            

            if pred_results is None:
                print(f"SQL execution error: {error}")
                return Message(role="user", content=error), 0.0
            else:
                df = pd.DataFrame(pred_results)
                res = df.to_string(index=False)
                if len(res) > 9000:
                    # just truncate
                    truncated_df = df.head(50)
                    res = "Truncated to 50 lines since returned response too long: " + truncated_df.to_string(
                        index=False
                    )  # or index=True if you want row numbers
                else:
                    res = "SQL execution results: " + res

                # print(f"SQL output: {res}")
                response = f"{res}\nYou have {MAX_TURNS - self.num_turn} turns left to complete the task."

                return Message(role="user", content=response), 0.0
        else:
            # no sql tool call means this is a final step
            is_valid, _, pred_sql, _ = verify_format_and_extract(action_text)

            if not is_valid:
                return Message(role="user", content="Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."), -1.0

            if self.dump_path is not None:
                with open(os.path.join(self.dump_path, f"{self.question_id}_pred.sql"), "w") as f:
                    f.write(pred_sql)

            pred = execute_sql_wrapper_single(self.db_file, pred_sql, self.timeout, action_text, self.db_modification_script)
            ref = execute_sql_wrapper_single(self.db_file, self.gold_answer, self.timeout, action_text, self.db_modification_script)

            _, _, pred_results, error, _ = pred
            _, _, gt_results, gt_error, _ = ref

            if pred_results is None:
                return None, 0.0
            elif gt_results is None:
                print(f"Ground truth SQL {self.question_id} execution error: {gt_error}. Possibly the gold SQL timed out.")
                return None, 0.0
            else:
                if grade(gt_results, pred_results, self.grading_method)[0]:
                    return None, 1.0
                else:
                    return None, 0.0


    def _is_done(self, action: str) -> bool:
        if self.num_turn == MAX_TURNS:
            return True
        return "<solution>" in action and "</solution>" in action


    async def step(self, action: Action) -> StepResult:
        self.num_turn += 1

        # step 1: parse the action tokens into a message
        # this step is specific to our library, but usually templated, so you can just copy it.
        (action_message, _parse_success) = self.renderer.parse_response(action)
        # print("=" * 100)
        # print(f"Action: {action_message['content']}")

        # step 2: based on the string answer, we compute the reward and the user turn.
        # This part is NOT templated, so you need to implement it. But it is plain python without using special libraries.
        user_turn, reward = self._get_user_turn(action_message["content"])
        # print(f"Next user turn: {user_turn}")
        # print(f"Current reward: {reward}")
        # print(f"Is done: {self._is_done(action_message['content'])}")
        # print(f"number of turns left: {MAX_TURNS - self.num_turn}")
        # print("=" * 100)

        # step 3: update the conversation history
        self.turns.append({"role": "assistant", "content": action_message["content"]})
        if user_turn is not None:
            self.turns.append(user_turn)
        episode_done = self._is_done(action_message['content'])

        if self._obs.length + self.max_output_tokens_per_turn > self.max_input_tokens:
            episode_done = True
            print("Observation too long, marking episode as done.")

        # step 4: return the step result
        step_result = StepResult(
            next_observation=self._obs,
            next_stop_condition=self.stop_condition,
            episode_done=episode_done,
            reward=reward
        )

        return step_result

class BIRDDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        data_path: str,
        db_path: str,
        db_modification_script_path: str,
        timeout: int,
        split: Literal["train", "test"] = "train",
        n_epochs: int = 1,
        num_data: int = -1,
        use_convo_prefix: bool = True,
        use_system_prompt: bool = True,
        max_output_tokens_per_turn: int = 3072,
        max_input_tokens: int = 32768,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.ds = cast(Dataset, load_dataset("parquet", data_files=data_path, keep_in_memory=True)["train"])
        # print two examples
        if split == "train":
            self.ds = self.ds.shuffle(seed=0)
            if num_data > 0:
                self.ds = self.ds.select(range(num_data))
            if n_epochs > 1:
                self.ds = concatenate_datasets([self.ds for _ in range(n_epochs)])
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.db_path = db_path
        self.timeout = timeout
        self.db_modification_script_path = db_modification_script_path
        self.dump_path = None
        self.use_system_prompt = use_system_prompt
        self.use_convo_prefix = use_convo_prefix
        self.max_output_tokens_per_turn = max_output_tokens_per_turn
        self.max_input_tokens = max_input_tokens

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

    def set_dump_path(self, dump_path: str) -> None:
        self.dump_path = dump_path

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem_id = f"{x['data_source']}_{x['question_id']}"
        problem = x["prompt"][1]["content"]
        answer = x["reward_spec"]["ground_truth"]
        grading_method = x["reward_spec"].get("grading_method", "multiset")
        dataset_name = x["data_source"]
        db_id = x["db_id"]
        db_file = f"{self.db_path}/{db_id}/{db_id}.sqlite"
        if os.path.exists(f"{self.db_modification_script_path}/{x['question_id']}.sql"):
            db_modification_script = f"{self.db_modification_script_path}/{x['question_id']}.sql"
        else:
            db_modification_script = None
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                SQLEnv, problem_id, problem, answer, grading_method, self.renderer, db_file, self.timeout, db_modification_script, self.dump_path, self.use_convo_prefix, self.use_system_prompt, self.max_output_tokens_per_turn, self.max_input_tokens
            ),
            num_envs=group_size,
            dataset_name=dataset_name,
        )


@chz.chz
class BIRDDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    renderer_name: str
    train_group_size: int
    base_url: str | None = None
    model_name: str
    data_path: str
    db_modification_script_path: str
    db_path: str
    timeout: int = 60
    add_noise: str | None = None
    n_epochs: int = 1
    num_data: int = -1
    use_convo_prefix: bool = True
    use_system_prompt: bool = True
    max_output_tokens_per_turn: int = 3072
    max_input_tokens: int = 32768

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        sql_renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))

        train_data_path = f"{self.data_path}/clean_train.parquet" 
        if "question" in self.add_noise and "sql" not in self.add_noise:
            train_data_path = f"{self.data_path}/noisy_question_train.parquet"
        if "sql" in self.add_noise and "question" not in self.add_noise:
            train_data_path = f"{self.data_path}/noisy_sql_train.parquet"
        if "question" in self.add_noise and "sql" in self.add_noise:
            train_data_path = f"{self.data_path}/noisy_train.parquet"
        if "db" in self.add_noise:
            assert "/databases" in self.db_path, "db_path must contain /databases if db noise is used"
            db_path = self.db_path.replace("/databases", "/original_databases")
        else:
            db_path = self.db_path

        test_data_path = f"{self.data_path}/combined_test.parquet"

        # log information about datasets
        logger.info(f"Training data path: {train_data_path}")
        logger.info(f"Test data path: {test_data_path}")
        logger.info(f"Database path: {db_path}")

        training_dataset = BIRDDataset(
            batch_size=self.batch_size,
            group_size=self.train_group_size,
            renderer=sql_renderer,
            data_path=train_data_path,
            db_modification_script_path=self.db_modification_script_path,
            timeout=self.timeout,
            db_path=db_path,
            split="train",
            n_epochs=self.n_epochs,
            num_data=self.num_data,
            use_system_prompt=self.use_system_prompt,
            use_convo_prefix=self.use_convo_prefix,
            max_output_tokens_per_turn=self.max_output_tokens_per_turn,
            max_input_tokens=self.max_input_tokens,
        )
        test_dataset = BIRDDataset(
            batch_size=self.batch_size,
            group_size=1,
            renderer=sql_renderer,
            data_path=test_data_path,
            db_modification_script_path=None,
            timeout=self.timeout,
            db_path=self.db_path,
            split="test",
            n_epochs=1,
            use_system_prompt=self.use_system_prompt,
            use_convo_prefix=self.use_convo_prefix,
            max_output_tokens_per_turn=self.max_output_tokens_per_turn,
            max_input_tokens=self.max_input_tokens,
        )
        return training_dataset, test_dataset

