#!/usr/bin/env python3
"""
One-time setup: creates the virtual environment venv/ and installs the packages.

    python setup_env.py

Run it again any time; an existing venv/ is reused and installed packages are kept.
"""

import os
import subprocess
import sys
import venv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT_DIR, "venv")

if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10 or newer is needed, this is {sys.version.split()[0]}")

if os.path.exists(VENV_DIR):
    print("venv/ already exists, reusing it")
else:
    print("creating venv/ ...")
    venv.create(VENV_DIR, with_pip=True)

# the venv's own python (Windows keeps it in Scripts/, macOS and Linux in bin/)
python = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python")
pip_install = [python, "-m", "pip", "install"]

# CPU-only PyTorch: this project never uses a GPU, and the CPU build is much smaller
print("installing PyTorch (CPU version) ...")
subprocess.run(pip_install + ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
print("installing the other packages ...")
subprocess.run(pip_install + ["-r", os.path.join(ROOT_DIR, "requirements.txt")], check=True)

print()
print("Setup complete. Next, activate the environment:")
print("  venv\\Scripts\\activate" if os.name == "nt" else "  source venv/bin/activate")
print("then run main.py")
