#!/usr/bin/env bash


start_server () {
  # Start the server
  echo "
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  Starting Streamlit UI...
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"
  uv pip install -e ".[ui]" --system
  streamlit run app/src/ui.py
}

start_server
