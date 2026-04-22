#!/bin/bash

# Create virtual environment
python3 -m venv my_env

# Activate it
source my_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# If an NVIDIA GPU is present, patch LD_LIBRARY_PATH so TensorFlow can find
# the CUDA libs bundled inside the venv by tensorflow[and-cuda]
if nvidia-smi &>/dev/null; then
    NVIDIA_LIB_PATHS=$(python3 -c "
import glob, os
paths = glob.glob('my_env/lib/python*/site-packages/nvidia/*/lib')
print(':'.join(os.path.abspath(p) for p in paths))
")
    ACTIVATE_FILE="my_env/bin/activate"
    if ! grep -q "nvidia" "$ACTIVATE_FILE"; then
        echo "" >> "$ACTIVATE_FILE"
        echo "# CUDA libs from tensorflow[and-cuda]" >> "$ACTIVATE_FILE"
        echo "export LD_LIBRARY_PATH=\"${NVIDIA_LIB_PATHS}:\$LD_LIBRARY_PATH\"" >> "$ACTIVATE_FILE"
        echo "GPU detected — LD_LIBRARY_PATH patched for CUDA."
    fi
else
    echo "No GPU detected — using CPU-only TensorFlow."
fi
