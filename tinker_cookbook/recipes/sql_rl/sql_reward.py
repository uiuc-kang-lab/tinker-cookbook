"""SQL reward computation for the text-to-SQL RL environment.

Ported from SkyRL's skyrl_gym/envs/sql/utils.py. The reward logic is:
  - Format violation (missing <think> or <solution> tags): -1.0
  - Correct SQL result set match: +1.0
  - Incorrect SQL result or execution error: 0.0
"""

from __future__ import annotations

import re
import sqlite3
import sys
import threading

from tinker_cookbook.renderers.base import Message, get_text_content

THINK_START = "<think>"
SOLUTION_START = "<solution>"
SOLUTION_END = "</solution>"


# ---------------------------------------------------------------------------
# Chat history reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_message_text(message: Message) -> str:
    """Reconstruct the full raw text of a message, re-inserting <think> tags.

    The Qwen3.5 renderer parses ``<think>`` blocks into structured
    ``ThinkingPart`` objects. This helper reverses that so the reward function
    can operate on a single flat string identical to what SkyRL produces.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for part in content:
        if part["type"] == "thinking":
            parts.append(f"<think>{part['thinking']}</think>")
        elif part["type"] == "text":
            parts.append(part["text"])
    return "".join(parts)


def build_chat_history_str(history: list[Message]) -> str:
    """Build a concatenated chat-history string that mirrors SkyRL's format.

    SkyRL concatenates ``[item["content"] for item in chat_history]`` where
    ``chat_history`` only contains the *response* turns (assistant messages and
    tool/observation messages), not the initial system + user prompt.

    In the tinker message list the first messages are the initial prompt
    (system, user).  We skip those and concatenate the rest:
    * **assistant** messages -> full text with ``<think>`` tags restored
    * **tool** messages -> content as-is (already wrapped in ``<observation>``)
    """
    parts: list[str] = []
    # Skip the initial prompt messages (system, user).
    prompt_done = False
    for msg in history:
        role = msg["role"]
        if not prompt_done:
            if role in ("system", "user"):
                # Last prompt message is always the user question.
                if role == "user":
                    prompt_done = True
                continue
            else:
                prompt_done = True
        if role == "assistant":
            parts.append(_reconstruct_message_text(msg))
        elif role == "tool":
            parts.append(get_text_content(msg))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Format verification  (ported from SkyRL utils.py)
# ---------------------------------------------------------------------------


def verify_format_and_extract(
    output: str,
) -> tuple[bool, list[str] | None, str | None]:
    """Verify output format and extract the predicted SQL from ``<solution>``.

    Checks (matching SkyRL exactly):
    1. Exactly one ``<solution>…</solution>`` pair.
    2. No forbidden tags (``think``, ``tool_call``, ``observation``) inside the
       solution text.
    3. At least one ``<think>…</think>`` block in the full output.
    4. After every ``</observation>``, a ``<think>`` block must follow.

    Returns ``(is_valid, thoughts, solution_sql)``.
    """
    if output.count(SOLUTION_START) != 1:
        return False, None, None
    pre_solution, tail = output.split(SOLUTION_START, 1)

    if tail.count(SOLUTION_END) != 1:
        return False, None, None
    solution_text, _ = tail.split(SOLUTION_END, 1)

    if re.search(r"</?(think|tool_call|observation)\b", solution_text, re.I):
        return False, None, None

    thoughts = re.findall(r"<think>(.*?)</think>", output, re.S)
    if not thoughts:
        return False, None, None

    for m in re.finditer(r"</observation>", pre_solution, re.I):
        rest = pre_solution[m.end() :].lstrip()
        if not rest.lower().startswith(THINK_START):
            return False, None, None

    return True, thoughts, solution_text.strip()


# ---------------------------------------------------------------------------
# SQL execution for reward grading  (ported from SkyRL utils.py)
# ---------------------------------------------------------------------------


def _execute_sql_for_reward(
    db_file: str, sql: str, timeout: int = 30
) -> tuple[frozenset | None, int]:
    """Execute SQL for reward comparison.  Returns ``(result_set, success)``."""
    res: tuple[frozenset | None, int] = (None, 0)
    done = threading.Event()
    conn_holder: dict[str, sqlite3.Connection | None] = {"conn": None}

    def worker() -> None:
        nonlocal res
        conn = None
        try:
            conn = sqlite3.connect(db_file, check_same_thread=False)
            conn_holder["conn"] = conn
            cur = conn.cursor()
            conn.execute("BEGIN TRANSACTION;")
            cur.execute(sql)
            rows = frozenset(cur.fetchall())
            conn.rollback()
            res = (rows, 1)
        except Exception:
            res = (None, 0)
        finally:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
            done.set()

    threading.Thread(target=worker, daemon=True).start()

    try:
        if not done.wait(timeout):
            conn = conn_holder.get("conn")
            if conn is not None:
                try:
                    conn.interrupt()
                except Exception:
                    pass
            done.wait()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        res = (None, 0)

    return res


def compute_score(completion: str, reference: str, db_file: str) -> float:
    """Compute reward score for a SQL completion.

    Returns:
        +1.0 for correct SQL (result sets match),
         0.0 for incorrect SQL or execution error,
        -1.0 for format violation.
    """
    try:
        is_valid, _, pred_sql = verify_format_and_extract(completion)
        if not is_valid or pred_sql is None:
            return -1.0

        pred_results, _ = _execute_sql_for_reward(db_file, pred_sql)
        gt_results, _ = _execute_sql_for_reward(db_file, reference)

        if (
            pred_results is not None
            and gt_results is not None
            and pred_results == gt_results
        ):
            return 1.0
        return 0.0
    except Exception as e:
        print(f"Unexpected error in reward computation: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Reward function factory (for AgentToolMessageEnv)
# ---------------------------------------------------------------------------


def make_sql_reward_fn(
    gold_sql: str, db_file: str
):
    """Create an async reward function compatible with ``AgentToolMessageEnv``.

    The returned callable has signature
    ``async (history: list[Message]) -> tuple[float, dict[str, float]]``.
    """

    async def reward_fn(
        history: list[Message],
    ) -> tuple[float, dict[str, float]]:
        chat_str = build_chat_history_str(history)
        reward = compute_score(chat_str, gold_sql, db_file)
        metrics: dict[str, float] = {
            "correct": float(reward > 0),
            "format_valid": float(reward >= 0),
        }
        return reward, metrics

    return reward_fn
