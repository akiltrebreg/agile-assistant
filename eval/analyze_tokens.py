"""Parse token usage from an eval log file and print distribution.

Usage:
    python -m eval.analyze_tokens eval_output_v15.txt
    python -m eval.analyze_tokens eval_output_v15.txt --by-category
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_TEST_RE = re.compile(r"\[(\d+)/\d+\]\s+#(\d+)")
_TOKENS_RE = re.compile(r"\[SQL Agent\] Tokens: prompt=(\d+), completion=(\d+), total=(\d+)")
_PROMPT_SIZE_RE = re.compile(r"\[SQL Agent\] Prompt size: chars=(\d+), messages=(\d+), mode=(\w+)")
_CACHE_RE = re.compile(r"\[SchemaLoader\] Cache (HIT|MISS|EXPIRED)")


def _percentile(data: list[int], pct: float) -> int:
    if not data:
        return 0
    s = sorted(data)
    k = int(len(s) * pct / 100)
    return s[min(k, len(s) - 1)]


def _stats(label: str, data: list[int]) -> None:
    if not data:
        print(f"  {label}: (no data)")
        return
    print(
        f"  {label}: n={len(data)} "
        f"p50={_percentile(data, 50)} "
        f"p95={_percentile(data, 95)} "
        f"max={max(data)} "
        f"mean={statistics.mean(data):.0f}"
    )


def _parse_log(lines: list[str]) -> dict:
    tokens = {"prompt": [], "completion": [], "total": []}
    prompt_chars: list[int] = []
    msg_counts: list[int] = []
    modes: dict[str, int] = defaultdict(int)
    per_test_totals: dict[int, int] = defaultdict(int)
    per_test_turns: dict[int, int] = defaultdict(int)
    cache_events: dict[str, int] = defaultdict(int)
    current: int | None = None

    for line in lines:
        if m := _TEST_RE.search(line):
            current = int(m.group(2))
        elif m := _TOKENS_RE.search(line):
            p, c, t = map(int, m.groups())
            tokens["prompt"].append(p)
            tokens["completion"].append(c)
            tokens["total"].append(t)
            if current is not None:
                per_test_totals[current] += t
                per_test_turns[current] += 1
        elif m := _PROMPT_SIZE_RE.search(line):
            prompt_chars.append(int(m.group(1)))
            msg_counts.append(int(m.group(2)))
            modes[m.group(3)] += 1
        elif m := _CACHE_RE.search(line):
            cache_events[m.group(1)] += 1

    return {
        "tokens": tokens,
        "prompt_chars": prompt_chars,
        "msg_counts": msg_counts,
        "modes": modes,
        "per_test_totals": per_test_totals,
        "per_test_turns": per_test_turns,
        "cache_events": cache_events,
    }


def _print_report(data: dict, name: str) -> None:
    tokens = data["tokens"]
    per_test_totals = data["per_test_totals"]
    per_test_turns = data["per_test_turns"]

    print(f"=== Token usage analysis: {name} ===\n")
    print(f"Turns observed: {len(tokens['prompt'])}")
    print(f"Tests observed: {len(per_test_totals)}\n")

    print("Per-turn tokens:")
    _stats("prompt_tokens     ", tokens["prompt"])
    _stats("completion_tokens ", tokens["completion"])
    _stats("total_tokens      ", tokens["total"])
    print("\nPer-turn prompt size (chars):")
    _stats("prompt_chars      ", data["prompt_chars"])
    _stats("msg_count         ", data["msg_counts"])
    print("\nPer-test aggregate tokens (sum across turns):")
    _stats("test_total_tokens ", list(per_test_totals.values()))
    _stats("test_turn_count   ", list(per_test_turns.values()))

    print("\nLLM mode split:")
    for mode, n in sorted(data["modes"].items()):
        print(f"  {mode}: {n}")

    print("\nSchema cache events:")
    for evt, n in sorted(data["cache_events"].items()):
        print(f"  {evt}: {n}")

    if tokens["prompt"]:
        ctx_limit = 8192
        max_prompt = max(tokens["prompt"])
        print(f"\nContext window usage (limit={ctx_limit}):")
        print(f"  max prompt = {max_prompt} ({max_prompt / ctx_limit * 100:.1f}%)")
        print(f"  headroom   = {ctx_limit - max_prompt} tokens")

    if per_test_totals:
        print("\nTop-10 tests by total tokens:")
        top = sorted(per_test_totals.items(), key=lambda kv: -kv[1])[:10]
        for tid, tok in top:
            print(f"  #{tid}: {tok} tokens, {per_test_turns[tid]} turns")


def analyze(path: Path) -> None:
    data = _parse_log(path.read_text().splitlines())
    _print_report(data, path.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_file", type=Path)
    args = ap.parse_args()
    if not args.log_file.exists():
        print(f"File not found: {args.log_file}", file=sys.stderr)
        return 1
    analyze(args.log_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
