"""
Grading utilities for the mix_general RL recipe.

Provides the same reward logic as SkyRL's mix_general env.py, ported to use
tinker-cookbook's math_grading primitives.
"""

import json
import logging

from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)

logger = logging.getLogger(__name__)

THOUGHT_DELIMITER_END = "</think>"


def _safe_grade_answer(given_answer: str, ground_truth: str, timeout: float = 10.0) -> bool:
    """grade_answer (sympy-based) with timeout protection."""
    out = run_with_timeout_signal(grade_answer, args=(given_answer, ground_truth), timeout_seconds=int(timeout))
    if out is None:
        return False
    return out


def _safe_grade_answer_math_verify(given_answer: str, ground_truth: str, timeout: float = 10.0) -> bool:
    """grade_answer_math_verify with timeout protection."""
    out = run_with_timeout_signal(grade_answer_math_verify, args=(given_answer, ground_truth), timeout_seconds=int(timeout))
    if out is None:
        return False
    return out


def grade(ground_truth: str, model_answer: str, method: str, timeout: float = 10.0) -> bool:
    """Grade a model answer against a ground truth using the specified method.

    Methods:
        - mqa / webinstruct-Multiple Choice: case-insensitive exact match, then math graders
        - legalbench: case-insensitive exact match
        - webinstruct-Float/Integer/Percentage/Fraction/Expression: math graders
        - webinstruct-String: case-insensitive exact match
        - webinstruct-List: set equality after comma-splitting
        - webinstruct-Boolean: semantic boolean match
    """
    valid_methods = {"mqa", "legalbench"} | {
        f"webinstruct-{t}"
        for t in ["Float", "Integer", "String", "Multiple Choice", "List", "Percentage", "Expression", "Boolean", "Fraction"]
    }
    if method not in valid_methods:
        raise ValueError(f"Unsupported grading method: {method}")

    if method in ("mqa", "webinstruct-Multiple Choice"):
        return (
            ground_truth.strip().lower() == model_answer.strip().lower()
            or _safe_grade_answer(model_answer, ground_truth, timeout)
            or _safe_grade_answer_math_verify(model_answer, ground_truth, timeout)
        )
    elif method == "legalbench":
        return ground_truth.strip().lower() == model_answer.strip().lower()
    elif method.startswith("webinstruct-"):
        answer_type = method.split("-", 1)[1]
        if answer_type in ("Float", "Integer", "Percentage", "Fraction", "Expression"):
            return (
                _safe_grade_answer(model_answer, ground_truth, timeout)
                or _safe_grade_answer_math_verify(model_answer, ground_truth, timeout)
            )
        elif answer_type == "String":
            return ground_truth.strip().lower() == model_answer.strip().lower()
        elif answer_type == "List":
            model_answers = {ans.strip().lower() for ans in model_answer.split(",")}
            ground_truths = {ans.strip().lower() for ans in json.loads(ground_truth)}
            return model_answers == ground_truths
        elif answer_type == "Boolean":
            positive = {"true", "yes", "1"}
            negative = {"false", "no", "0"}
            m = model_answer.strip().lower()
            g = ground_truth.strip().lower()
            if g in positive:
                return m in positive
            elif g in negative:
                return m in negative
            else:
                return m == g
        else:
            raise ValueError(f"Unsupported answer type for webinstruct: {answer_type}")
    else:
        raise ValueError(f"Unsupported grading method: {method}")
