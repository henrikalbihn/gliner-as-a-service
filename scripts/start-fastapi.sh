#!/usr/bin/env bash

WORKER_CLASS=uvicorn.workers.UvicornWorker
N_WORKERS=${N_GUNICORN_WORKERS:-1}
PORT=${API_PORT:-8080}
ADDRESS="0.0.0.0:${PORT}"


start_server () {
  # Start the server
  echo "
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  [${N_WORKERS}x ${WORKER_CLASS}] workers listening @ ${ADDRESS}...
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"

  gunicorn -w ${N_WORKERS} -k ${WORKER_CLASS} \
    app.main:app \
    --bind ${ADDRESS}
}

start_server
