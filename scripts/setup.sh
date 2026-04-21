#!/bin/bash

# Deactivate conda base if active so only (my_env) shows in prompt
if type conda &>/dev/null; then
    conda deactivate 2>/dev/null || true
fi

# Create virtual environment
python3 -m venv my_env
source my_env/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Enter into the virtual environment
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exec "$SHELL"
fi

# If GPU is available, install CUDA libraries and patch the activate script
if type nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "GPU detected — installing CUDA libraries..."
    pip install "tensorflow[and-cuda]"
    cat >> my_env/bin/activate << 'CUDAEOF'

export LD_LIBRARY_PATH=$(python -c "
import os, nvidia
p = os.path.dirname(nvidia.__file__)
libs = [os.path.join(p, d, 'lib') for d in os.listdir(p) if os.path.isdir(os.path.join(p, d, 'lib'))]
print(':'.join(libs))
" 2>/dev/null):${LD_LIBRARY_PATH}
CUDAEOF
    source my_env/bin/activate
    echo "GPU setup complete."
else
    echo "No GPU detected — using CPU only."
fi

# Verify environment
echo "--- Environment Check ---"
which python
