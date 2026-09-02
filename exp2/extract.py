from __future__ import annotations

import ast
import hashlib
import re

EXTRACTION_RULES_VERSION = "exp2-extract-v2"     # covers extraction AND assembly

PYTHON_TAGS = {"python", "py", "python3", "py3"}

FENCE_RE = re.compile(
    r"(?:^|\n)[ \t]*```([A-Za-z0-9_+.#-]*)(?:[ \t][^\n]*)?\n"
    r"(?P<body>.*?)"
    r"(?:(?P<close>\n[ \t]*```[ \t]*(?=\n|$))|\Z)",
    re.DOTALL,
)

RULES = (
    "1. python-tagged fence that defines the entry point; "
    "2. first python-tagged fence; 3. first untagged fence; "
    "(an unclosed fence counts, its body runs to the end of the text); "
    "4. no fence: the whole response if it parses as Python; "
    "otherwise the sample is unparseable and dropped (that sample, all arms)."
)


def find_fences(text: str) -> list[dict]:
    fences = []
    for m in FENCE_RE.finditer(text or ""):
        fences.append({"tag": (m.group(1) or "").lower(), "body": m.group("body"),
                       "closed": m.group("close") is not None})
    return fences


def defines(code: str, entry_point: str | None) -> bool:
    if not entry_point:
        return False
    return re.search(rf"^[ \t]*def[ \t]+{re.escape(entry_point)}[ \t]*\(", code, re.M) is not None


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def extract_code(text: str, entry_point: str | None = None) -> tuple[str | None, dict]:
    """(code, info).  code is None when nothing usable could be extracted."""
    fences = find_fences(text)
    python = [f for f in fences if f["tag"] in PYTHON_TAGS]
    untagged = [f for f in fences if f["tag"] == ""]

    chosen, method = None, "none"
    for f in python:
        if defines(f["body"], entry_point):
            chosen, method = f, "fence_python_entry"
            break
    if chosen is None and python:
        chosen, method = python[0], "fence_python"
    if chosen is None and untagged:
        chosen, method = untagged[0], "fence_untagged"

    info = {"method": method, "n_fences": len(fences), "closed": None,
            "rules": EXTRACTION_RULES_VERSION}
    if chosen is not None:
        code = chosen["body"].strip("\n")
        info["closed"] = chosen["closed"]
        return code, info

    stripped = (text or "").strip()
    if stripped and parses(stripped):
        info["method"] = "raw_parses"
        return stripped, info
    return None, info


ASSEMBLY_RULES = (
    "humaneval: import lines found inside the prompt's function bodies are hoisted to "
    "module level; then the prompt itself (its imports and helper functions stay in "
    "scope; the entry-point stub is docstring-only); then the code -- appended directly "
    "when it is a body-only completion, after a blank line when it defines the entry "
    "point, whose def then overrides the stub -- then test + check(entry_point); "
    "mbpp: test_imports + code + asserts"
)
_INDENTED_IMPORT = re.compile(r"^[ \t]+((?:import|from)[ \t]+\S.*)$")


def build_program(problem: dict, code: str) -> tuple[str, str]:
    if problem["source"] == "humaneval":
        prompt = problem["prompt"]
        hoisted = [m.group(1).strip() for m in map(_INDENTED_IMPORT.match, prompt.splitlines())
                   if m]
        header = "\n".join(hoisted) + "\n\n" if hoisted else ""
        if defines(code, problem["entry_point"]):
            body, mode = prompt.rstrip("\n") + "\n\n\n" + code, "prompt_then_code"
        else:
            body, mode = prompt + code, "prompt+code"
        program = (header + body + "\n\n" + problem["test"]
                   + f"\n\ncheck({problem['entry_point']})\n")
        return program, mode
    if problem["source"] == "mbpp":
        header = "\n".join(problem.get("test_imports") or [])
        program = header + "\n\n" + code + "\n\n" + "\n".join(problem["test_list"]) + "\n"
        return program, "code"
    raise ValueError(f"unknown problem source {problem['source']!r}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
