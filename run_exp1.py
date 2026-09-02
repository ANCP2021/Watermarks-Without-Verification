#!/usr/bin/env python3
"""Experiment 1 - does SynthID-Text watermarking degrade prose quality?

One resumable pipeline, run once per generator model. 

    python run_exp1.py generate --model google/gemma-2-9b-it --out runs/gemma2-9b
    python run_exp1.py generate --model meta-llama/Llama-3.1-8B-Instruct --out runs/llama31-8b

    # scoring - independent, rerunnable, order-free
    python run_exp1.py ppl    --out runs/gemma2-9b   # --rescore to redo an existing ppl.jsonl
    python run_exp1.py sim    --out runs/gemma2-9b
    python run_exp1.py judge  --out runs/gemma2-9b
    python run_exp1.py report --out runs/gemma2-9b

    python run_exp1.py all --model Qwen/Qwen2.5-0.5B-Instruct --out runs/smoke --smoke
"""

from __future__ import annotations

import argparse
import sys

from exp1.common import DEFAULT_PROMPTS_PATH, DEFAULT_WATERMARK_CONFIG_PATH, log


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True,
                        help="run directory; all artefacts live here")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH),
                        help="shared prompts.json (created once, then reloaded)")
    parser.add_argument("--watermark-config", default=str(DEFAULT_WATERMARK_CONFIG_PATH),
                        help="shared SynthID key file (created once, never regenerated)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="auto = bfloat16 on GPU, float32 on CPU")
    parser.add_argument("--smoke", action="store_true",
                        help="wiring test: 10 prompts, shorter outputs, small scorers")
    parser.add_argument("--force", action="store_true",
                        help="resume even if run parameters changed (mixes records)")


def _generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="generator model id")
    parser.add_argument("--revision", default=None, help="pin the model to a commit sha")
    parser.add_argument("--attn-impl", default=None,
                        help="eager|sdpa|flash_attention_2 (default: eager for gemma-2)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="default 512, or 128 under --smoke")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of prompts (debugging)")


def _ppl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ppl-model", default=None, help="override the scorer LM")
    parser.add_argument("--ppl-max-length", type=int, default=2048)
    parser.add_argument("--rescore", action="store_true", help="archive ppl.jsonl and score every prompt again (needed when the scoring definition changed)")


def _sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sim-model", default=None, help="override the embedding model")


def _judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge-model", default=None, help="override the judge LM")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_exp1.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prompts", help="create prompts.json (done automatically otherwise)")
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH))
    p.add_argument("--dolly-revision", default=None)

    p = sub.add_parser("keys", help="create watermark_config.json (done automatically otherwise)")
    p.add_argument("--watermark-config", default=str(DEFAULT_WATERMARK_CONFIG_PATH))

    p = sub.add_parser("generate", help="produce arms A, B1, B2")
    _common(p); _generate_args(p)

    p = sub.add_parser("ppl", help="mean per-token NLL of each response, "
                                   "conditioned on its prompt")
    _common(p); _ppl_args(p)

    p = sub.add_parser("sim", help="embedding cosine similarity")
    _common(p); _sim_args(p)

    p = sub.add_parser("judge", help="blinded pairwise preference, positions swapped")
    _common(p); _judge_args(p)

    p = sub.add_parser("report", help="arm-parity check, then results.md + "
                                      "results.csv + diagnostics.md + pass/fail")
    _common(p)

    p = sub.add_parser("all", help="generate, then every scoring stage, then report")
    _common(p); _generate_args(p); _ppl_args(p); _sim_args(p); _judge_args(p)

    return parser


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prompts":
        from exp1.prompts import load_prompts
        load_prompts(args.prompts, revision=args.dolly_revision)
        return 0

    if args.command == "keys":
        from exp1.watermark_setup import ensure_watermark_config
        ensure_watermark_config(args.watermark_config)
        return 0

    from exp1.generate import run_generate
    from exp1.report import run_report
    from exp1.scoring_judge import run_judge
    from exp1.scoring_ppl import run_ppl
    from exp1.scoring_sim import run_sim

    stages = {"generate": run_generate, "ppl": run_ppl, "sim": run_sim,
              "judge": run_judge, "report": run_report}

    if args.command == "all":
        for name in ("generate", "ppl", "sim", "judge", "report"):
            log(f"=== stage: {name} ===")
            stages[name](args)
            _free_gpu()
        return 0

    stages[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
