"""Response Agent evaluation runner.

Loads response_golden_dataset.json and runs each `state` through
ResponseAgent.process(), then validates `final_response` against
`checks` (must_contain / must_not_contain / length / language /
expected_branch / sources).

Placeholders ``"<<GENERATE: team=lpop, limit=70>>"`` in `state.sql_result`
are expanded at load time via a SQL query against the live database.

Usage:
    docker compose run --rm --no-deps app \\
        python -m eval.run_response_eval --experiment response_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN = _EVAL_DIR / "response_golden_dataset.json"
_RESULTS_DIR = _EVAL_DIR / "results"

_PLACEHOLDER_RE = re.compile(r"^<<GENERATE:\s*(.+?)\s*>>$")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


# ── Placeholder expansion ────────────────────────────────────


def _parse_placeholder(spec: str) -> dict[str, str]:
    """Parse 'team=lpop, limit=70, type=Bug' → {team: lpop, limit: 70, type: Bug}."""
    out: dict[str, str] = {}
    for part in spec.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _build_query(params: dict[str, str]) -> tuple[str, dict[str, Any]]:
    """Build SELECT * FROM report_agile_dashboard with WHERE clause from params."""
    where: list[str] = []
    bind: dict[str, Any] = {}
    if "team" in params:
        where.append("feature_teams ILIKE :team")
        bind["team"] = f"%{params['team']}%"
    if "type" in params:
        where.append("issue_type ILIKE :type")
        bind["type"] = f"%{params['type']}%"
    if "status" in params:
        where.append("issue_status_act ILIKE :status")
        bind["status"] = f"%{params['status']}%"

    where_sql = " AND ".join(where) if where else "TRUE"
    limit = int(params.get("limit", 100))
    sql = f"SELECT * FROM report_agile_dashboard WHERE {where_sql} LIMIT {limit}"
    return sql, bind


def _expand_placeholder(spec_str: str, db: Any) -> list[dict[str, Any]]:
    """Expand a <<GENERATE: ...>> placeholder by querying the DB."""
    m = _PLACEHOLDER_RE.match(spec_str)
    if not m:
        raise ValueError(f"Bad placeholder format: {spec_str}")
    params = _parse_placeholder(m.group(1))
    sql, bind = _build_query(params)
    logger.info("[Response Eval] Expanding placeholder: %s → %s", spec_str, sql)
    rows = db.execute_query(sql, bind)
    logger.info("[Response Eval] Got %d rows for %s", len(rows), spec_str)
    return rows


def _maybe_expand_state(state: dict[str, Any], db: Any | None) -> dict[str, Any]:
    """If sql_result is a placeholder string, replace with rows from DB."""
    sr = state.get("sql_result")
    if not isinstance(sr, str):
        return state
    if not _PLACEHOLDER_RE.match(sr):
        return state
    if db is None:
        raise RuntimeError(
            f"Cannot expand placeholder {sr!r}: DB unavailable. Start postgres or check connection."
        )
    state["sql_result"] = _expand_placeholder(sr, db)
    return state


# ── Check evaluation ─────────────────────────────────────────


def _check_must_contain(response: str, terms: list[str]) -> list[str]:
    """Return list of MISSING terms (case-insensitive substring match)."""
    if not terms:
        return []
    r = response.lower()
    return [t for t in terms if t.lower() not in r]


def _check_must_contain_any(response: str, groups: list[list[str]]) -> list[list[str]]:
    """For each OR-group: pass if ANY term matches. Returns failed groups."""
    if not groups:
        return []
    r = response.lower()
    failed = []
    for group in groups:
        if not any(t.lower() in r for t in group):
            failed.append(group)
    return failed


def _check_must_not_contain(response: str, terms: list[str]) -> list[str]:
    """Return list of FORBIDDEN terms that appeared (case-insensitive)."""
    if not terms:
        return []
    r = response.lower()
    return [t for t in terms if t.lower() in r]


def _check_language_ru(response: str) -> bool:
    """Heuristic: at least 30% of letter chars are Cyrillic."""
    if not response:
        return False
    letters = [c for c in response if c.isalpha()]
    if not letters:
        return False
    cyrillic = len(_CYRILLIC_RE.findall(response))
    return cyrillic / len(letters) >= 0.3


def _check_sources(response: str, must_have: bool) -> bool:
    """Check presence/absence of 'Источники' block."""
    has = "Источники" in response
    return has == must_have


def _evaluate_checks(response: str, checks: dict[str, Any]) -> dict[str, Any]:
    """Run all checks and return per-check result + overall pass/fail."""
    missing = _check_must_contain(response, checks.get("must_contain", []))
    missing_any = _check_must_contain_any(response, checks.get("must_contain_any", []))
    forbidden = _check_must_not_contain(response, checks.get("must_not_contain", []))

    lang_ok = True
    if checks.get("language") == "ru":
        lang_ok = _check_language_ru(response)

    length = len(response)
    min_ok = length >= checks.get("min_length", 0)
    max_ok = length <= checks.get("max_length", 10**9)

    sources_ok = True
    if "must_have_sources" in checks:
        sources_ok = _check_sources(response, checks["must_have_sources"])

    all_ok = (
        not missing
        and not missing_any
        and not forbidden
        and lang_ok
        and min_ok
        and max_ok
        and sources_ok
    )

    return {
        "pass": all_ok,
        "missing_terms": missing,
        "missing_any_groups": missing_any,
        "forbidden_terms": forbidden,
        "language_ok": lang_ok,
        "length": length,
        "length_ok": min_ok and max_ok,
        "sources_ok": sources_ok,
    }


# ── Pipeline ─────────────────────────────────────────────────


def _build_response_agent() -> Any:
    """Build ResponseAgent with main LLM client."""
    from hse_prom_prog.agents.response_agent import ResponseAgent
    from hse_prom_prog.llm.client import LLMClient

    client = LLMClient()
    return ResponseAgent(client)


def _try_get_db() -> Any | None:
    """Get DB connection or None if unavailable."""
    try:
        from hse_prom_prog.database.connection import get_database

        db = get_database()
        if db.test_connection():
            logger.info("[Response Eval] DB connection ready")
            return db
        logger.warning("[Response Eval] DB connection test failed")
    except Exception as e:
        logger.warning("[Response Eval] DB unavailable: %s", e)
    return None


def _format_check_detail(check_result: dict, err: str | None) -> str:
    """Build a short failure-detail string for log output."""
    bits = []
    if check_result.get("missing_terms"):
        bits.append(f"missing={check_result['missing_terms']}")
    if check_result.get("missing_any_groups"):
        bits.append(f"missing_any={check_result['missing_any_groups']}")
    if check_result.get("forbidden_terms"):
        bits.append(f"forbidden={check_result['forbidden_terms']}")
    if not check_result.get("language_ok", True):
        bits.append("not_ru")
    if not check_result.get("length_ok", True):
        bits.append(f"len={check_result['length']}")
    if not check_result.get("sources_ok", True):
        bits.append("sources_mismatch")
    if err:
        bits.append(f"err={err[:50]}")
    return "; ".join(bits) if bits else "all checks pass"


def _run_single_case(case: dict, response_agent: Any, db: Any | None) -> dict:
    """Run one test case through the agent and return a result record."""
    cid = case["id"]
    name = case["name"]
    category = case["category"]
    checks = case["checks"]

    try:
        state = _maybe_expand_state(dict(case["state"]), db)
    except Exception as e:
        logger.error("[Response Eval] Placeholder expansion failed for #%d: %s", cid, e)
        return {
            "id": cid,
            "name": name,
            "category": category,
            "expected_branch": checks.get("expected_branch"),
            "response": "",
            "checks": {"pass": False, "error": f"placeholder: {e}"},
            "latency_s": 0,
            "error": str(e),
        }

    t0 = time.perf_counter()
    try:
        output = response_agent.process(state)
        response = output.get("final_response", "")
        err = None
    except Exception as e:
        response = ""
        err = str(e)
        logger.error("[Response Eval] Error on #%d: %s", cid, e)
    latency = time.perf_counter() - t0

    check_result = _evaluate_checks(response, checks)
    if err:
        check_result["pass"] = False
        check_result["error"] = err

    return {
        "id": cid,
        "name": name,
        "category": category,
        "expected_branch": checks.get("expected_branch"),
        "response": response,
        "checks": check_result,
        "latency_s": round(latency, 3),
        "error": err,
    }


def _run_eval(response_agent: Any, dataset: list[dict], db: Any | None) -> list[dict]:
    results = []
    total = len(dataset)
    for i, case in enumerate(dataset, 1):
        logger.info(
            "[%d/%d] #%d [%s] %s", i, total, case["id"], case["category"], case["name"][:60]
        )
        record = _run_single_case(case, response_agent, db)
        results.append(record)
        status = "PASS" if record["checks"].get("pass") else "FAIL"
        detail = _format_check_detail(record["checks"], record["error"])
        logger.info("  %s: %s (%.2fs)", status, detail[:120], record["latency_s"])
    return results


# ── Reporting ────────────────────────────────────────────────


def _print_per_category(results: list[dict]) -> None:
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)
    print("\nPer-category:")
    rows = []
    for cat, items in sorted(categories.items()):
        n = len(items)
        ok = sum(1 for x in items if x["checks"].get("pass"))
        rows.append([cat, n, f"{ok}/{n}", f"{ok / n * 100:.0f}%"])
    print(tabulate(rows, headers=["Category", "N", "Passed", "Rate"], tablefmt="simple"))


def _failure_tags(record: dict) -> str:
    c = record["checks"]
    tags = []
    if c.get("missing_terms"):
        tags.append(f"MISSING={c['missing_terms']}")
    if c.get("missing_any_groups"):
        tags.append(f"MISSING_ANY={c['missing_any_groups']}")
    if c.get("forbidden_terms"):
        tags.append(f"FORBIDDEN={c['forbidden_terms']}")
    if not c.get("language_ok", True):
        tags.append("LANG")
    if not c.get("length_ok", True):
        tags.append(f"LEN={c['length']}")
    if not c.get("sources_ok", True):
        tags.append("SRC")
    if record["error"]:
        tags.append(f"ERR={record['error'][:40]}")
    return " | ".join(tags) if tags else "?"


def _print_failures(results: list[dict]) -> None:
    failures = [r for r in results if not r["checks"].get("pass")]
    if not failures:
        return
    print(f"\nFailures ({len(failures)}):")
    for r in failures:
        name_short = r["name"] if len(r["name"]) <= 50 else r["name"][:47] + "..."
        print(f"  #{r['id']} [{r['category']}] {name_short}")
        print(f"      {_failure_tags(r)}")


def _print_latency(results: list[dict]) -> None:
    lats = [r["latency_s"] for r in results if not r["error"]]
    if not lats:
        return
    s = sorted(lats)
    p50 = s[len(s) // 2]
    p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
    print(f"\nLatency: n={len(lats)}, p50={p50:.2f}s, p95={p95:.2f}s, max={max(lats):.2f}s")


def _print_check_breakdown(results: list[dict]) -> None:
    print("\nCheck-type breakdown:")
    cb = [
        ["missing_terms", sum(1 for r in results if r["checks"].get("missing_terms"))],
        ["forbidden_terms", sum(1 for r in results if r["checks"].get("forbidden_terms"))],
        ["language", sum(1 for r in results if not r["checks"].get("language_ok", True))],
        ["length", sum(1 for r in results if not r["checks"].get("length_ok", True))],
        ["sources", sum(1 for r in results if not r["checks"].get("sources_ok", True))],
    ]
    print(tabulate(cb, headers=["Check", "Failed cases"], tablefmt="simple"))


def _print_summary(results: list[dict], experiment: str) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["checks"].get("pass"))

    print(f"\n{'=' * 60}")
    print(f"Response Eval: {experiment}  ({total} cases)")
    print(f"{'=' * 60}\n")
    print(f"  Passed : {passed}/{total} ({passed / total * 100:.1f}%)")
    print(f"  Failed : {total - passed}/{total}")

    _print_per_category(results)
    _print_failures(results)
    _print_latency(results)
    _print_check_breakdown(results)
    print()


# ── Main ─────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Response Agent evaluation")
    parser.add_argument("--experiment", default="response_baseline")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    with _GOLDEN.open() as f:
        dataset = json.load(f)
    logger.info("Loaded %d Response test cases", len(dataset))

    db = _try_get_db()
    response_agent = _build_response_agent()

    results = _run_eval(response_agent, dataset, db)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    total = len(results)
    passed = sum(1 for r in results if r["checks"].get("pass"))

    output = {
        "experiment": args.experiment,
        "timestamp": timestamp,
        "aggregate": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 3) if total else 0,
        },
        "results": results,
    }

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / f"{args.experiment}_{timestamp}.json"
    with out_path.open("w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Saved results to %s", out_path)

    _print_summary(results, args.experiment)


if __name__ == "__main__":
    main()
