# ── Wait for Postgres to be ready, then run uvicorn ──
# Usage: docker-entrypoint.sh [timeout_seconds]
# Default timeout: 30 seconds

set -e

HOST="${HERMES_DB_HOST:-postgres}"
PORT="${HERMES_DB_PORT:-5432}"
TIMEOUT="${1:-30}"

echo "Waiting for Postgres at ${HOST}:${PORT} (timeout=${TIMEOUT}s)..."

for i in $(seq 1 "${TIMEOUT}"); do
    if nc -z "${HOST}" "${PORT}" 2>/dev/null; then
        echo "Postgres is ready."
        break
    fi
    if [ "${i}" -eq "${TIMEOUT}" ]; then
        echo "ERROR: Postgres did not become ready within ${TIMEOUT}s."
        exit 1
    fi
    sleep 1
done

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
