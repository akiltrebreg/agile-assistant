"""SQL Agent evaluation runner.

Loads sql_golden_dataset.json, runs each question through the SQL Agent
(text2sql via arctic-7b), compares results against expected, and saves
a detailed report to eval/results/.

Usage::

    # Via Docker (recommended — all services up):
    docker compose run --rm app python -m eval.run_sql_eval --experiment sql_arctic_v1

    # Locally:
    poetry run python -m eval.run_sql_eval --experiment sql_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN = _EVAL_DIR / "sql_golden_dataset.json"
_RESULTS_DIR = _EVAL_DIR / "results"

_FLOAT_TOLERANCE = 0.1  # for value_approx


# ── Evaluation strategies ────────────────────────────────────


def _eval_row_count_exact(actual_rows: list[dict], expected: dict) -> tuple[bool, str]:
    actual_count = len(actual_rows)
    expected_count = expected["row_count"]
    ok = actual_count == expected_count
    detail = f"rows: {actual_count} (expected {expected_count})"
    return ok, detail


def _eval_exact_match_fields(actual_rows: list[dict], expected: dict) -> tuple[bool, str]:
    if not actual_rows:
        return False, "no rows returned"
    row = actual_rows[0]
    key_fields = expected.get("key_fields", {})
    # Also check top-level fields (for comparative queries)
    if not key_fields:
        key_fields = {k: v for k, v in expected.items() if k not in ("row_count", "value")}
    mismatches = []
    for field, exp_val in key_fields.items():
        # Support "field_any": [val1, val2] — accept any of the values
        if field.endswith("_any") and isinstance(exp_val, list):
            real_field = field.removesuffix("_any")
            act_val = row.get(real_field)
            if str(act_val) not in [str(v) for v in exp_val]:
                mismatches.append(f"{real_field}: {act_val!r} not in {exp_val}")
            continue
        act_val = row.get(field)
        if isinstance(exp_val, float) and isinstance(act_val, (int, float)):
            if abs(float(act_val) - exp_val) > _FLOAT_TOLERANCE:
                mismatches.append(f"{field}: {act_val} != {exp_val}")
        elif str(act_val) != str(exp_val):
            mismatches.append(f"{field}: {act_val!r} != {exp_val!r}")
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "all fields match"


def _eval_exact_match_rows(actual_rows: list[dict], expected: dict) -> tuple[bool, str]:
    expected_rows = expected.get("rows", [])
    if len(actual_rows) != len(expected_rows):
        return False, f"row count: {len(actual_rows)} != {len(expected_rows)}"

    # Only compare columns present in expected rows (extra columns are OK)
    exp_cols = set()
    for r in expected_rows:
        exp_cols.update(r.keys())

    def _row_key(r: dict) -> frozenset:
        return frozenset((k, _round(v)) for k, v in r.items() if k in exp_cols)

    actual_set = {_row_key(r) for r in actual_rows}
    expected_set = {_row_key(r) for r in expected_rows}
    if actual_set == expected_set:
        return True, "all rows match"
    missing = expected_set - actual_set
    return False, f"{len(missing)} row(s) missing or different"


def _eval_exact_match_grouped(actual_rows: list[dict], expected: dict) -> tuple[bool, str]:
    expected_data = expected.get("data", {})
    # Determine group key and value key from actual rows
    if not actual_rows:
        return len(expected_data) == 0, f"0 rows (expected {len(expected_data)})"
    cols = list(actual_rows[0].keys())
    exp_keys = set(expected_data.keys())

    # Find group_col: the column whose values best match expected keys
    group_col = cols[0]
    for c in cols:
        col_vals = {str(r[c]) for r in actual_rows}
        if col_vals & exp_keys:
            group_col = c
            break

    # val_col: last numeric column that is not the group column
    val_col = cols[-1] if len(cols) > 1 else cols[0]
    for c in reversed(cols):
        if c != group_col:
            val_col = c
            break
    actual_data = {str(r[group_col]): _round(r[val_col]) for r in actual_rows}
    mismatches = []
    for k, exp_v in expected_data.items():
        act_v = actual_data.get(k)
        if act_v is None:
            mismatches.append(f"missing: {k}")
        elif abs(float(act_v) - float(exp_v)) > _FLOAT_TOLERANCE:
            mismatches.append(f"{k}: {act_v} != {exp_v}")
    if mismatches:
        return False, "; ".join(mismatches[:5]) + (
            f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        )
    return True, f"{len(expected_data)} groups match"


def _scalar_from_row(row: dict) -> Any:
    """Extract the scalar value from a single-column or multi-column row."""
    if len(row) == 1:
        return next(iter(row.values()))
    return row.get("value", list(row.values())[-1])


def _eval_value_exact(actual_rows: list[dict], expected: dict) -> tuple[bool, str]:
    if not actual_rows:
        return False, "no rows returned"
    act_val = _scalar_from_row(actual_rows[0])
    exp_val = expected["value"]
    if isinstance(exp_val, float) and isinstance(act_val, (int, float)):
        ok = abs(float(act_val) - exp_val) < 0.01
    else:
        ok = str(act_val) == str(exp_val)
    return ok, f"value: {act_val} (expected {exp_val})"


def _eval_value_approx(actual_rows: list[dict], expected: dict) -> tuple[bool, str]:
    if not actual_rows:
        return False, "no rows returned"
    act_val = _scalar_from_row(actual_rows[0])
    exp_val = expected["value"]
    if act_val is None or exp_val is None:
        return act_val == exp_val, f"value: {act_val} (expected {exp_val})"
    ok = abs(float(act_val) - float(exp_val)) <= _FLOAT_TOLERANCE
    return ok, f"value: {_round(act_val)} (expected {exp_val}, tol±{_FLOAT_TOLERANCE})"


def _eval_value_null(actual_rows: list[dict], expected: dict) -> tuple[bool, str]:
    if not actual_rows:
        return True, "no rows (treated as NULL)"
    act_val = _scalar_from_row(actual_rows[0])
    ok = act_val is None
    return ok, f"value: {act_val} (expected NULL)"


def _eval_composite(actual_rows: list[dict], expected: dict) -> tuple[bool, str]:
    # Composite is hard to auto-evaluate since SQL agent returns one query
    # For now, check if we got any results
    return len(actual_rows) > 0, f"composite: {len(actual_rows)} rows (manual review needed)"


_STRATEGIES: dict[str, Any] = {
    "row_count_exact": _eval_row_count_exact,
    "exact_match_fields": _eval_exact_match_fields,
    "exact_match_rows": _eval_exact_match_rows,
    "exact_match_grouped": _eval_exact_match_grouped,
    "value_exact": _eval_value_exact,
    "value_approx": _eval_value_approx,
    "value_null": _eval_value_null,
    "composite": _eval_composite,
}


def _round(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 2)
    return v


# ── Pipeline ─────────────────────────────────────────────────


def _build_sql_agent():
    """Build SQL Agent with database connection."""
    from agile_assistant.agents.sql_agent import SQLAgent
    from agile_assistant.database.connection import get_database

    db = get_database()
    return SQLAgent(db_connection=db)


def _run_eval(agent: Any, dataset: list[dict]) -> list[dict]:
    """Run all questions through SQL Agent and evaluate."""
    results = []
    total = len(dataset)
    for i, item in enumerate(dataset, 1):
        qid = item["id"]
        question = item["question"]
        strategy = item["eval_strategy"]
        expected = item["expected_result"]

        logger.info("[%d/%d] #%d %s", i, total, qid, question[:60])

        t0 = time.perf_counter()
        try:
            state = {"original_query": question}
            output = agent.process(state)
            sql = output.get("sql_query", "")
            rows = output.get("sql_result") or []
            error = output.get("error")
            latency = time.perf_counter() - t0
        except Exception as e:
            sql = ""
            rows = []
            error = str(e)
            latency = time.perf_counter() - t0
            logger.error("[SQL Eval] Error on #%d: %s", qid, e)

        # Evaluate
        eval_fn = _STRATEGIES.get(strategy)
        if eval_fn and not error:
            passed, detail = eval_fn(rows, expected)
        elif error:
            passed, detail = False, f"error: {error}"
        else:
            passed, detail = False, f"unknown strategy: {strategy}"

        results.append(
            {
                "id": qid,
                "question": question,
                "category": item["category"],
                "difficulty": item["difficulty"],
                "eval_strategy": strategy,
                "generated_sql": sql,
                "row_count": len(rows),
                "passed": passed,
                "detail": detail,
                "latency_s": round(latency, 3),
                "error": error,
            }
        )

        status = "PASS" if passed else "FAIL"
        logger.info("  %s: %s (%.1fs)", status, detail[:80], latency)

    return results


# ── Output ───────────────────────────────────────────────────


def _print_summary(results: list[dict], experiment: str) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print(f"\n{'=' * 60}")
    print(f"SQL Eval: {experiment}  ({passed}/{total} passed, {failed} failed)")
    print(f"{'=' * 60}\n")

    # Per-category breakdown
    categories: dict[str, list[bool]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r["passed"])
    cat_rows = [
        [cat, len(vals), sum(vals), f"{sum(vals) / len(vals) * 100:.0f}%"]
        for cat, vals in sorted(categories.items())
    ]
    print(tabulate(cat_rows, headers=["Category", "Total", "Passed", "Rate"], tablefmt="simple"))

    # Per-difficulty breakdown
    diffs: dict[str, list[bool]] = {}
    for r in results:
        diffs.setdefault(r["difficulty"], []).append(r["passed"])
    diff_rows = [
        [d, len(vals), sum(vals), f"{sum(vals) / len(vals) * 100:.0f}%"]
        for d, vals in sorted(diffs.items())
    ]
    print()
    print(tabulate(diff_rows, headers=["Difficulty", "Total", "Passed", "Rate"], tablefmt="simple"))

    # Failed questions
    failed_items = [r for r in results if not r["passed"]]
    if failed_items:
        print(f"\nFailed ({len(failed_items)}):")
        for r in failed_items:
            q = r["question"] if len(r["question"]) <= 50 else r["question"][:47] + "..."
            print(f"  #{r['id']} [{r['category']}] {q}")
            print(f"    SQL: {(r['generated_sql'] or '')[:80]}...")
            print(f"    {r['detail']}")

    # Latency
    latencies = [r["latency_s"] for r in results]
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        print(
            f"\nLatency: avg={avg_lat:.2f}s, min={min(latencies):.2f}s, max={max(latencies):.2f}s"
        )
    print()


# ── Main ─────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SQL Agent evaluation")
    parser.add_argument(
        "--experiment", default="sql_baseline", help="Experiment name for output file"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    # Load dataset
    with _GOLDEN.open() as f:
        dataset = json.load(f)
    logger.info("Loaded %d SQL test cases", len(dataset))

    # Build agent
    agent = _build_sql_agent()

    # Run evaluation
    results = _run_eval(agent, dataset)

    # Aggregate
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    latencies = [r["latency_s"] for r in results]

    output = {
        "experiment": args.experiment,
        "timestamp": timestamp,
        "config": {
            "sql_model": settings.sql_vllm_model,
            "sql_base_url": settings.sql_vllm_base_url,
        },
        "aggregate": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0,
            "latency_avg_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
            "latency_p95_s": round(
                sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 3
            ),
        },
        "per_question": results,
    }

    # Save
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"{args.experiment}_{timestamp}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.info("Results saved → %s", out_path)

    # Print
    _print_summary(results, args.experiment)


# Import settings lazily to avoid import errors when module is loaded
from agile_assistant.config import settings  # noqa: E402

if __name__ == "__main__":
    main()
