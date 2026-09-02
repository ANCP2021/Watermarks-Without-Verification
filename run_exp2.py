#!/usr/bin/env python3
"""Experiment 2 - SynthID-Text on code: functional correctness and detectability
from the same outputs.

One resumable pipeline, run once per generator model.

    python run_exp2.py generate --model google/gemma-2-9b-it             --out runs/exp2-gemma2-9b
    python run_exp2.py generate --model meta-llama/Llama-3.1-8B-Instruct --out runs/exp2-llama31-8b

    python run_exp2.py execute --out runs/exp2-gemma2-9b   # sandboxed reference tests
    python run_exp2.py detect  --out runs/exp2-gemma2-9b   # mean g-value, code + Exp 1 prose
    python run_exp2.py report  --out runs/exp2-gemma2-9b

    python run_exp2.py all --model Qwen/Qwen2.5-0.5B-Instruct --out runs/exp2-smoke --smoke

    python run_exp2.py problems     # write problems.json (done automatically otherwise)
    python run_exp2.py selftest     # canonical solutions through the harness: all must pass
    python run_exp2.py dump-table   # persist the SynthID g-value table (run on the cuda box)
"""

from __future__ import annotations

import argparse
import sys

from exp1.common import DEFAULT_WATERMARK_CONFIG_PATH, log
from exp2.problems import DEFAULT_PROBLEMS_PATH
from exp2.settings import EXEC_MEM_MB, EXEC_TIMEOUT_S, EXEC_WORKERS


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="run directory; all artefacts live here")
    parser.add_argument("--problems", default=str(DEFAULT_PROBLEMS_PATH),
                        help="shared problems.json (created once, then reloaded)")
    parser.add_argument("--watermark-config", default=str(DEFAULT_WATERMARK_CONFIG_PATH),
                        help="shared SynthID key file (created once, never regenerated)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="auto = bfloat16 on GPU, float32 on CPU")
    parser.add_argument("--smoke", action="store_true",
                        help="wiring test: 5 problems (3 HumanEval + 2 MBPP), 256 new tokens")
    parser.add_argument("--force", action="store_true",
                        help="resume even if run parameters changed (mixes records)")


def _generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="generator model id")
    parser.add_argument("--revision", default=None, help="pin the model to a commit sha")
    parser.add_argument("--attn-impl", default=None,
                        help="eager|sdpa|flash_attention_2 (default: eager for gemma-2)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="default 512, or 256 under --smoke")
    parser.add_argument("--samples", type=int, default=None,
                        help="samples per problem per arm (default 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of problems (debugging)")


def _execute_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeout", type=float, default=EXEC_TIMEOUT_S,
                        help="seconds per candidate before the process group is killed")
    parser.add_argument("--mem-mb", type=int, default=EXEC_MEM_MB,
                        help="address-space cap per candidate (RLIMIT_AS; not enforced on macOS)")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"concurrent sandboxes (default {EXEC_WORKERS})")
    parser.add_argument("--exec-prefix", default=None,
                        help="command prefix for the sandbox, e.g. 'unshare -rn' on Linux "
                             "for a real network namespace")


def _detect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exp1-runs", default="runs/*",
                        help="glob of Exp 1 run directories whose prose is scored when "
                             "their generator model matches this run's")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_exp2.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("problems", help="create problems.json (done automatically otherwise)")
    p.add_argument("--problems", default=str(DEFAULT_PROBLEMS_PATH))
    p.add_argument("--humaneval-revision", default=None)
    p.add_argument("--mbpp-revision", default=None)

    p = sub.add_parser("selftest", help="run every canonical solution through the harness")
    p.add_argument("--problems", default=str(DEFAULT_PROBLEMS_PATH))
    _execute_args(p)

    p = sub.add_parser("dump-table", help="draw the SynthID sampling table exactly as generation "
                                          "did (cuda; seed and size from watermark_config.json) "
                                          "and save sampling_table.pt next to it. `detect` "
                                          "refuses to run without that file")
    p.add_argument("--watermark-config", default=str(DEFAULT_WATERMARK_CONFIG_PATH))
    p.add_argument("--device", default="cuda", choices=["cuda", "emulate-cuda"],
                   help="cuda = the real draw (default); emulate-cuda = numpy Philox "
                        "re-implementation of the cuda stream, provisional until confirmed")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing sampling_table.pt that holds a different table")

    p = sub.add_parser("generate", help="produce arms A, B1, B2 (n samples each)")
    _common(p); _generate_args(p)

    p = sub.add_parser("execute", help="run every sample against its reference tests "
                                       "in an isolated subprocess")
    _common(p); _execute_args(p)

    p = sub.add_parser("detect", help="mean g-value of every code sample and of the "
                                      "matching Exp 1 prose, with prefix curves")
    _common(p); _detect_args(p)

    p = sub.add_parser("report", help="parity check, then results.md/csv, metrics.json, "
                                      "curves.csv and the detection figure")
    _common(p)

    p = sub.add_parser("all", help="generate, execute, detect, report")
    _common(p); _generate_args(p); _execute_args(p); _detect_args(p)
    return parser


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "problems":
        from exp2.problems import load_problems
        load_problems(args.problems, humaneval_revision=args.humaneval_revision,
                      mbpp_revision=args.mbpp_revision)
        return 0

    if args.command == "selftest":
        from exp2.selftest import run_selftest
        return run_selftest(args)

    if args.command == "dump-table":
        from exp1.watermark_setup import dump_sampling_table
        return dump_sampling_table(args)

    from exp2.detect import run_detect
    from exp2.execute import run_execute
    from exp2.generate import run_generate
    from exp2.report import run_report

    stages = {"generate": run_generate, "execute": run_execute,
              "detect": run_detect, "report": run_report}

    if args.command == "all":
        for name in ("generate", "execute", "detect", "report"):
            log(f"=== stage: {name} ===")
            stages[name](args)
            _free_gpu()
        return 0

    stages[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
