#!/usr/bin/env bash

set -e

REQS_FILE_IN=dependencies/requirements-in.txt
REQS_FILE_OUT=dependencies/requirements-out.txt
REQS_FILE_CLEAN=dependencies/requirements-clean.txt
REQS_FILE_DEV=dependencies/requirements-dev.txt

pip_step () {
  # Upgrade pip
  pip install --upgrade pip -q
  # Install:
  #   - uv: a rust-based pip drop-in replacement
  pip install --upgrade uv -q
}

uv_step () {
  echo "Creating virtual environment..."
  # Create a virtual environment
  uv venv .venv --python $(cat .python-version)
  echo "Adding dependencies to uv / pyproject.toml..."
  # Add to uv / pyproject.toml
  cat ${REQS_FILE_IN} | xargs uv add
  echo "Adding UI dependencies..."
  # Add the UI dependencies
  # Join contents of dependencies/requirements-ui.txt into a single space separated string
  UI_DEPS=$(cat dependencies/requirements-ui.txt | tr '\n' ' ')
  uv add --optional ui ${UI_DEPS}
  echo "Adding dev dependencies..."
  # Add the dev dependencies
  cat ${REQS_FILE_DEV} | xargs uv add --dev

  echo "Compiling version-locked dependencies..."
  # Compile the version-locked dependencies
  uv pip compile ${REQS_FILE_IN} -o ${REQS_FILE_OUT}
  echo "Syncing virtual environment with version-locked dependencies..."
  # Sync the virtual environment with the version-locked dependencies
  uv pip sync ${REQS_FILE_OUT}
  echo "Getting only the non-commented lines..."
  # Get only the non-commented lines (aka skip the ones starting with '# ')
  cat ${REQS_FILE_OUT} | grep -v '# ' > ${REQS_FILE_CLEAN}
}

main () {
  echo "Setting up Python environment..."
  pip_step
  uv_step
  echo "Done!"
}

main
