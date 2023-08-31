#!/bin/bash

VENV_NAME=".venv"
REQUIREMENTS_FILE="requirements.txt"
PYTHON_SCRIPT="src/models/sweep.py"

# Install pip if not already installed
if ! command -v pip &> /dev/null; then
	echo "Installing pip..."
	python3 -m ensurepip --default-pip
fi

# Create and activate virtual environment
if [ ! -d "$VENV_NAME" ]; then
	echo "Creating virtual env..."
	python3 -m venv "$VENV_NAME"
fi
source "$VENV_NAME"/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r "$REQUIREMENTS_FILE"

# Run Python script
python "$PYTHON_SCRIPT"
