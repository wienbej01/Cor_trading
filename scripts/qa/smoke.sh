#!/bin/bash
set -e

echo "Installing project in editable mode..."
pip install -e .

echo "Installing dev dependencies..."
pip install -r requirements-dev.txt

echo "Running pytest..."
pytest -q

echo "Running ruff..."
ruff check .

echo "Running mypy..."
mypy src