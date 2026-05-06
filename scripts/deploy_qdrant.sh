#!/usr/bin/env bash
set -euo pipefail

# Поднимает Qdrant в Docker с персистентным томом под атомарные утверждения:
# файлы коллекций и снапшоты лежат в data/qdrant (или QDRANT_DATA_DIR).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${ROOT}/docker-compose.qdrant.yml"

QDRANT_SERVICE="${QDRANT_SERVICE:-qdrant}"
HOST_PORT="${QDRANT_HOST_PORT:-6333}"
HOST_URL="http://127.0.0.1:${HOST_PORT}"

# Абсолютный путь к данным по умолчанию (можно переопределить QDRANT_DATA_DIR).
DEFAULT_DATA="${ROOT}/data/qdrant"
DATA_DIR="${QDRANT_DATA_DIR:-${DEFAULT_DATA}}"

if ! command -v docker &>/dev/null; then
  echo "Не найден docker в PATH." >&2
  exit 1
fi

compose() {
  # QDRANT_DATA_DIR для compose задаём относительно корня проекта только если пользователь её не переопределил.
  export QDRANT_DATA_DIR="${DATA_DIR}"
  docker compose -f "${COMPOSE_FILE}" "$@"
}

mkdir -p "${DATA_DIR}"

echo "Запуск Qdrant (compose), данные: ${DATA_DIR}"
compose up -d "${QDRANT_SERVICE}"

echo "Ожидание API ${HOST_URL}…"
until curl -sf "${HOST_URL}/" >/dev/null 2>&1; do
  sleep 1
done

echo
echo "Готово. REST: ${HOST_URL}"
echo "Для клиента из кода задаёте например:"
echo "  export QDRANT_URL=${HOST_URL}"
echo
echo "Проверка:"
echo "  curl -s ${HOST_URL}/collections | head"
