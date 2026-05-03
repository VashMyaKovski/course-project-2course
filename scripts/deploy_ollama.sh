#!/usr/bin/env bash
set -euo pipefail

# Поднимает Ollama в Docker и скачивает Meta Llama 3.1 8B из каталога Ollama (тег llama3.1:8b).
# После успеха выставьте переменные (скрипт выведет блок export) или передавайте их в приложении.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${ROOT}/docker-compose.ollama.yml"

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
OLLAMA_SERVICE="${OLLAMA_SERVICE:-ollama}"
HOST_PORT="${OLLAMA_HOST_PORT:-11434}"
HOST_URL="http://127.0.0.1:${HOST_PORT}"

if ! command -v docker &>/dev/null; then
  echo "Не найден docker в PATH." >&2
  exit 1
fi

compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

echo "Запуск Ollama (compose)…"
compose up -d "${OLLAMA_SERVICE}"

echo "Ожидание API ${HOST_URL}…"
until curl -sf "${HOST_URL}/api/version" >/dev/null 2>&1; do
  sleep 2
done

echo "Скачивание модели '${OLLAMA_MODEL}' (может занять много времени и гигабайты трафика)…"
compose exec -T "${OLLAMA_SERVICE}" ollama pull "${OLLAMA_MODEL}"

echo
echo "Готово. Для fact-extractor задайте окружение:"
echo "  export LLAMA_CHAT_API_BASE=${HOST_URL}/v1"
echo "  export LLAMA_CHAT_API_KEY=ollama"
echo "  export LLAMA_CHAT_MODEL=${OLLAMA_MODEL}"
echo
echo "Проверка chat completions:"
echo "  curl -s ${HOST_URL}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' -H 'Authorization: Bearer ollama' \\"
echo "    -d '{\"model\":\"${OLLAMA_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}'"
