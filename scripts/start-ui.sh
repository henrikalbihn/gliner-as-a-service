#!/usr/bin/env bash


start_server () {
  # Start the server
  echo "
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  Starting Streamlit UI...
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"
  source .venv/bin/activate
  uv pip install -r dependencies/requirements-ui.txt
  streamlit run app/src/ui.py
}

start_server
