#!/usr/bin/env bash

PROMPT="This is a story all about how my life got flipped turned upside down and I'd like to take a minute just sit right there I'll tell you how I became the prince of a town called Bel-Air."

HOST="http://localhost:8080"


JSON_DATA=$(cat <<EOF
{
  "inputs": [
    "${PROMPT}"
  ],
  "labels": [
    "PERSON",
    "PLACE",
    "THING",
    "ORGANIZATION",
    "DATE",
    "TIME"
  ]
}
EOF
)

test_predict() {
  echo "Testing predict API with the following data:"
  echo "${JSON_DATA}" | jq
  export RESPONSE=$(curl -X 'POST' \
    "${HOST}/predict" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d "${JSON_DATA}")
  # {
  #   "task_id": "f6d07a3a-10de-42c1-8dfb-8bf1ac9de0d9",
  #   "status": "Processing"
  # }
  export TASK_ID=$(echo ${RESPONSE} | jq .task_id | tr -d '"')
  if [ -z "${TASK_ID}" ]; then
    echo "Task ID is empty"
    exit 1
  fi
  echo "
  Task ID: [${TASK_ID}]
  "
}

poll_till_complete() {
  sleep 2
  RESPONSE=$(curl -X 'GET' \
    "${HOST}/result/${TASK_ID}" \
    -H 'accept: application/json')
  STATUS=$(echo ${RESPONSE} | jq .status | tr -d '"')

  # if processing, sleep 5 seconds and try again
  while [ "${STATUS}" == "Processing" ]; do
    echo "Task is still processing..."
    sleep 5
    RESPONSE=$(curl -X 'GET' \
      "${HOST}/result/${TASK_ID}" \
      -H 'accept: application/json')
    STATUS=$(echo ${RESPONSE} | jq .status | tr -d '"')
  done
  echo "
  Task [${TASK_ID}] status: [${STATUS}]!
  "
  FMT_RESP=$(echo ${RESPONSE} | jq .result.predictions)
  echo ${FMT_RESP} | jq
  # {
  #   "task_id": "f6d07a3a-10de-42c1-8dfb-8bf1ac9de0d9",
  #   "status": "Success",
  #   "result": {
  #     "predictions": [
  #       [
  #         {
  #           "start": 16,
  #           "end": 20,
  #           "text": "John",
  #           "label": "PERSON",
  #           "score": 0.995429277420044
  #         }
  #       ]
  #     ]
  #   }
  # }
}

main() {
  test_predict
  poll_till_complete
}

main
