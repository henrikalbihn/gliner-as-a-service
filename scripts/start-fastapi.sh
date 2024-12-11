#!/usr/bin/env bash

API_PORT=${API_PORT:-8000}
API_HOST=${API_HOST:-0.0.0.0}

main () {
  if [ ! -d ".venv" ]; then
    uv venv .venv
    uv sync
  fi
  source .venv/bin/activate
  echo "
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  FastAPI server listening @ ${API_HOST}:${API_PORT}...
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"
  fastapi run app/main.py \
    --reload \
    --host ${API_HOST} \
    --port ${API_PORT}
}

main
