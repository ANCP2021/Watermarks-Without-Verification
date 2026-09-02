from __future__ import annotations

from .settings import HUMANEVAL_TEMPLATE, INSTRUCTION, MBPP_TEMPLATE


def user_message(problem: dict) -> str:
    if problem["source"] == "humaneval":
        return HUMANEVAL_TEMPLATE.format(instruction=INSTRUCTION,
                                         prompt=problem["prompt"].rstrip("\n"))
    if problem["source"] == "mbpp":
        tests = "\n".join(list(problem.get("test_imports") or []) + list(problem["test_list"]))
        return MBPP_TEMPLATE.format(instruction=INSTRUCTION,
                                    prompt=problem["prompt"].strip(), tests=tests)
    raise ValueError(f"unknown problem source {problem['source']!r}")
