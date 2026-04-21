#!/bin/bash

# Deactivate the virtual environment
# Only works if executed with 'source scripts/remove.sh'
if type deactivate &>/dev/null; then
    deactivate
    echo "Virtual environment deactivated."
else
    echo "Note: No active virtual environment found in this shell."
fi

# Remove the virtual environment
if [ -d "my_env" ]; then
    rm -rf my_env
    echo "Removed 'my_env' directory."
fi

# Remove trained model and split dataset
if [ -d "models/splited" ]; then
    rm -rf models/splited
    echo "Removed 'models/splited' directory."
fi

# Remove augmented dataset
if [ -d "dataset/augmented" ]; then
    rm -rf dataset/augmented
    echo "Removed 'dataset/augmented' directory."
fi

# Remove TensorBoard logs
if [ -d "logs" ]; then
    rm -rf logs
    echo "Removed 'logs' directory."
fi

# Remove pytest cache
if [ -d ".pytest_cache" ]; then
    rm -rf .pytest_cache
    echo "Removed '.pytest_cache' directory."
fi

# Remove Python cache files recursively
find . -type d -name "__pycache__" -not -path "./my_env/*" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -not -path "./my_env/*" -delete 2>/dev/null
echo "Removed __pycache__ and .pyc files."

echo "Cleanup complete!"