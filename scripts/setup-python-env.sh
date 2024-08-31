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
  #   - poetry: a python dependency manager
  # We're using both because uv is MUCH faster for local development
  # Poetry creates the lock file and is used in production
  pip install uv poetry -q
}

uv_step () {
  # Create a virtual environment
  uv venv .venv
  # Activate the virtual environment
  source .venv/bin/activate
  # Install the dependencies
  uv pip install -r ${REQS_FILE_IN}
  # Compile the version-locked dependencies
  uv pip compile ${REQS_FILE_IN} -o ${REQS_FILE_OUT}
  # Sync the virtual environment with the version-locked dependencies
  uv pip sync ${REQS_FILE_OUT}
  # Get only the non-commented lines (aka skip the ones starting with '# ')
  cat ${REQS_FILE_OUT} | grep -v '# ' > ${REQS_FILE_CLEAN}
}

poetry_step () {
  # Update the poetry pyproject.toml file
  cat ${REQS_FILE_CLEAN} | xargs poetry add
  # Update the poetry pyproject.toml file with the dev dependencies
  cat ${REQS_FILE_DEV} | xargs poetry add --group dev
  # Generate the poetry.lock file
  poetry lock
}

install_pkg () {
  # Install a package
  uv pip install -e .
}

main () {
  pip_step
  uv_step
  poetry_step
  install_pkg
}

main
