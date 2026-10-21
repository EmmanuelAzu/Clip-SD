#!/usr/bin/env bash
# Environment setup for Clip-SD on mscluster.
# Run this from inside the repo directory: bash setup_env.sh
set -uo pipefail

echo "=================================================="
echo " Environment Setup: Clip-SD"
echo "=================================================="

# ---- Step 1: Try to find and load a Python module (best-effort) ----
PYTHON_MODULE_LOADED=false

echo "[*] Searching for a Python module..."
if command -v module &> /dev/null; then
    for candidate in "python/3.10" "Python/3.10" "python/3.10.4" "Python/3.10.4" \
                      "python3" "Python" "python" "anaconda3" "Anaconda3" "miniconda3"; do
        if module load "$candidate" 2>/dev/null; then
            echo "[+] Loaded module: $candidate"
            PYTHON_MODULE_LOADED=true
            break
        fi
    done
else
    echo "[!] 'module' command not found on this system -- skipping module load."
fi

if [ "$PYTHON_MODULE_LOADED" = false ]; then
    echo "[!] No Python module could be auto-loaded from the common names tried."
    echo "    Falling back to whatever python3 is already on PATH..."
fi

# ---- Step 2: Confirm python3 is actually available now, one way or another ----
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "[X] ERROR: python3 is not available even after module attempts."
    echo "    Run this manually and share the output so we can pick the right module:"
    echo "      module avail 2>&1 | grep -i python"
    echo "      module avail 2>&1 | grep -i conda"
    exit 1
fi

echo "[*] Using: $(which python3)"
echo "[*] Version: $(python3 --version)"

PY_MAJOR_MINOR=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[*] Detected Python $PY_MAJOR_MINOR"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "[+] Python version OK (>= 3.9)."
else
    echo "[!] WARNING: Python < 3.9 detected. torch>=2.0.0 may not have a compatible wheel."
fi

# ---- Step 3: Build the virtual environment ----
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment at ./$VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "[*] Virtual environment already exists at ./$VENV_DIR -- reusing it."
fi

source "$VENV_DIR/bin/activate"
echo "[+] Activated venv: $(which python3)"

# ---- Step 4: Upgrade pip and install requirements ----
echo ""
echo "[*] Installing requirements (needs network access -- run this on the login"
echo "    node if compute nodes are firewalled off from the internet)..."
pip install --upgrade pip
pip install -r requirements.txt

# ---- Step 5: Verify the install, especially GPU/CUDA ----
echo ""
echo "=================================================="
echo " Verification"
echo "=================================================="
python3 -c "
import sys
print('Python:', sys.version.split()[0])
try:
    import torch
    print('Torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA version:', torch.version.cuda)
        print('GPU device:', torch.cuda.get_device_name(0))
    else:
        print('[!] No GPU detected by torch -- pipeline will run on CPU.')
        print('    If you expected a GPU here, you likely need a CUDA-specific torch')
        print('    build, e.g.: pip install torch --index-url https://download.pytorch.org/whl/cu121')
except ImportError as e:
    print('[X] torch import failed:', e)
"

python3 -c "import clip; print('[+] CLIP import: OK')" 2>&1 \
    || echo "[X] CLIP import failed -- check the 'git+https://...' install step above for errors."

python3 -c "
import pandas, numpy, sklearn, matplotlib, seaborn, PIL, tqdm
print('[+] pandas, numpy, sklearn, matplotlib, seaborn, PIL, tqdm: all OK')
" 2>&1 || echo "[X] One or more supporting packages failed to import -- check pip install output above."

echo ""
echo "=================================================="
echo "[+] Setup complete."
echo "    Activate this environment in future sessions with:"
echo "      source venv/bin/activate"
echo "=================================================="
