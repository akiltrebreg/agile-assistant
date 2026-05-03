#!/usr/bin/env bash
# Wrapper для Locust-прогонов: создаёт папку со временной меткой,
# складывает в неё CSV / HTML / log и заготовку notes.md.
#
# Usage:
#   ./loadtest/run.sh <name> [-- доп.флаги для locust]
#
# Примеры:
#   ./loadtest/run.sh smoke-1u    --users 1  --spawn-rate 1 --run-time 2m
#   ./loadtest/run.sh baseline-5u --users 5  --spawn-rate 1 --run-time 10m
#   ./loadtest/run.sh capacity    --users 10 --spawn-rate 1 --run-time 15m
#   ./loadtest/run.sh stress      --users 50 --spawn-rate 1 --run-time 15m
#
# Override host:
#   LOCUST_HOST=http://localhost:80 ./loadtest/run.sh smoke-1u --users 1 ...

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run-name> [locust flags...]" >&2
    echo "Example: $0 smoke-1u --users 1 --spawn-rate 1 --run-time 2m" >&2
    exit 1
fi

RUN_NAME="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${LOCUST_HOST:-http://195.209.218.21}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M)"
RUN_DIR="$REPO_ROOT/loadtest/results/${TIMESTAMP}_${RUN_NAME}"

mkdir -p "$RUN_DIR"

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a)"
GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"

cat >"$RUN_DIR/notes.md" <<EOF
# Run: $RUN_NAME

- **Старт**: $(date -Iseconds)
- **Хост**: $HOST
- **Locust args**: $*
- **Git commit**: $GIT_COMMIT
- **Git branch**: $GIT_BRANCH

## Итоги

(заполнить после прогона — итоговые цифры из \`run_stats.csv\`)

## Наблюдения

(аномалии, корреляции с Grafana, всё необычное)
EOF

echo ">>> Run dir:  $RUN_DIR"
echo ">>> Host:     $HOST"
echo ">>> Args:     $*"
echo ">>> Starting Locust..."
echo

locust -f "$REPO_ROOT/loadtest/locustfile.py" \
    --host "$HOST" \
    --headless \
    --csv "$RUN_DIR/run" \
    --html "$RUN_DIR/report.html" \
    --logfile "$RUN_DIR/locust.log" \
    "$@"

echo
echo ">>> Run complete. Artefacts:"
ls -la "$RUN_DIR"
