"""Compare two or more RAG evaluation experiments.

Reads result JSON files produced by ``run_eval.py`` and prints a
side-by-side metric comparison table with deltas highlighted.

Usage::

    python -m eval.compare eval/results/baseline_*.json eval/results/v2_*.json
    python -m eval.compare results/a.json results/b.json results/c.json
"""

import argparse
import json
import logging
from pathlib import Path

from tabulate import tabulate

logger = logging.getLogger(__name__)

# ANSI colours for terminal output
_GREEN = "\033[92m"
_RED = "\033[91m"
_RESET = "\033[0m"


# ── helpers ──────────────────────────────────────────────────


def _load_result(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        msg = f"File not found: {p}"
        raise FileNotFoundError(msg)
    with p.open() as f:
        return json.load(f)


def _format_delta(delta: float) -> str:
    """Format delta with colour and sign."""
    sign = "+" if delta >= 0 else ""
    raw = f"{sign}{delta:.4f}"
    if delta > 0:
        return f"{_GREEN}{raw}{_RESET}"
    if delta < 0:
        return f"{_RED}{raw}{_RESET}"
    return raw


def _label(result: dict) -> str:
    """Short human-readable label for an experiment."""
    name = result.get("experiment", "?")
    ts = result.get("timestamp", "")
    return f"{name} ({ts})" if ts else name


# ── two-experiment comparison ────────────────────────────────


def _compare_pair(results: list[dict]) -> None:
    """Print metric table with delta between first and last experiment."""
    base, *rest = results
    last = rest[-1]

    base_agg = base["aggregate"]
    last_agg = last["aggregate"]
    all_metrics = list(dict.fromkeys([*base_agg, *last_agg]))

    headers = ["Metric", _label(base)]
    for r in rest:
        headers.append(_label(r))
    headers.append("delta (last − first)")

    rows = []
    for m in all_metrics:
        row: list[str] = [m]
        b_val = base_agg.get(m)
        row.append(f"{b_val:.4f}" if b_val is not None else "—")
        for r in rest:
            v = r["aggregate"].get(m)
            row.append(f"{v:.4f}" if v is not None else "—")
        l_val = last_agg.get(m)
        if b_val is not None and l_val is not None:
            row.append(_format_delta(l_val - b_val))
        else:
            row.append("—")
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="simple"))


# ── config diff ──────────────────────────────────────────────


def _print_config_diff(results: list[dict]) -> None:
    """Show config parameters that differ between experiments."""
    configs = [r.get("config", {}) for r in results]
    labels = [_label(r) for r in results]
    all_keys = list(dict.fromkeys(k for c in configs for k in c))

    diff_rows = []
    for key in all_keys:
        values = [c.get(key, "—") for c in configs]
        if len(set(str(v) for v in values)) > 1:
            diff_rows.append([key, *[str(v) for v in values]])

    if not diff_rows:
        print("\nPipeline configs are identical across experiments.")
        return

    print("\nConfig differences:")
    print(tabulate(diff_rows, headers=["param", *labels], tablefmt="simple"))


# ── main ─────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare RAG evaluation experiments",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Two or more result JSON files from eval/results/",
    )
    args = parser.parse_args(argv)

    if len(args.files) < 2:
        parser.error("Need at least 2 result files to compare.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = [_load_result(f) for f in args.files]
    logger.info("Comparing %d experiments\n", len(results))

    _compare_pair(results)
    _print_config_diff(results)
    print()


if __name__ == "__main__":
    main()
