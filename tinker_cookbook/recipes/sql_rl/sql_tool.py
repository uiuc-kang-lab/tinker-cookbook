"""SQL execution tool for the text-to-SQL RL environment.

Ported from SkyRL's skyrl_gym/tools/sql.py to tinker-cookbook's @tool interface.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import threading
from typing import Annotated

import pandas as pd

from tinker_cookbook.tool_use.tools import simple_tool_result, tool
from tinker_cookbook.tool_use.types import ToolResult

MAX_OUTPUT_CHARS = 9000


def _execute_sql_with_timeout(db_file: str, sql: str, timeout: int = 5) -> str:
    """Execute a SQL query with a timeout using a worker thread and conn.interrupt().

    Ported from SkyRL's SQLCodeExecutorToolGroup.sql().

    Returns:
        Result string (formatted DataFrame or error message).
    """
    res_holder: dict[str, object] = {"value": None}
    done = threading.Event()
    conn_holder: dict[str, sqlite3.Connection | None] = {"conn": None}

    def worker() -> None:
        conn = None
        try:
            conn = sqlite3.connect(db_file, check_same_thread=False)
            conn_holder["conn"] = conn
            cursor = conn.cursor()
            conn.execute("BEGIN TRANSACTION;")
            cursor.execute(sql)
            execution_res = frozenset(cursor.fetchall())
            conn.rollback()
            res_holder["value"] = execution_res
        except Exception as e:
            res_holder["value"] = f"Error executing SQL: {e}, db file: {db_file}"
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

    timeout_occurred = False
    try:
        if not done.wait(timeout):
            timeout_occurred = True
            conn = conn_holder.get("conn")
            if conn is not None:
                try:
                    conn.interrupt()
                except Exception:
                    pass
            done.wait()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        res_holder["value"] = str(e)

    if timeout_occurred:
        return f"SQL Timeout:\n{sql}"

    res = res_holder["value"]
    if isinstance(res, frozenset):
        df = pd.DataFrame(res)
        result = df.to_string(index=False)
        if len(result) > MAX_OUTPUT_CHARS:
            truncated_df = df.head(50)
            result = (
                "Truncated to 50 lines since returned response too long: "
                + truncated_df.to_string(index=False)
            )
        return result
    else:
        return str(res)


class SQLExecutorTools:
    """SQL execution tools for a specific database.

    Each environment instance should get its own SQLExecutorTools so that the
    per-instance turn counter is accurate.
    """

    def __init__(self, db_file: str, max_turns: int):
        self.db_file = db_file
        self.max_turns = max_turns
        self._calls = 0

    @tool
    async def execute_sql(
        self,
        query: Annotated[str, "The SQL query to execute against the database"],
    ) -> ToolResult:
        """Execute a SQL query against the database and return the results. The returned dataframe will be truncated to 50 rows if the result is too long."""
        self._calls += 1
        turns_left = self.max_turns - self._calls

        obs = await asyncio.to_thread(
            _execute_sql_with_timeout, self.db_file, query, 5
        )
        reminder = (
            f"<reminder>You have {turns_left} turns left to complete the task.</reminder>"
        )
        content = f"\n\n<observation>{obs}\n{reminder}</observation>\n\n"
        return simple_tool_result(content)
