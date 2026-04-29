"""Launch M5.6 Persona Runtime (Streamlit app).

Usage:
    streamlit run scripts/run_m56_local.py
    # or
    py scripts/run_m56_local.py

Options are passed through to streamlit:
    streamlit run scripts/run_m56_local.py -- --storage artifacts/m56_personas/ --port 8501
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# streamlit run expects the app file as its argument.
# We point it at the app module inside the runtime package.
_APP_PATH = _project_root / "segmentum" / "dialogue" / "runtime" / "app.py"

if __name__ == "__main__":
    import subprocess

    cmd = [sys.executable, "-m", "streamlit", "run", str(_APP_PATH)]
    # Forward any extra args
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd)
