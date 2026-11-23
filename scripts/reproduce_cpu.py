"""Rebuild the deployable baseline and regression evidence; never runs GPU work or final test."""

import os
import subprocess
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])
for arguments in [
    ["prepare"],
    ["train-baseline"],
    ["evaluate-baseline", "--split", "gate"],
    ["gate"],
]:
    subprocess.run([sys.executable, "-m", "b77.cli", *arguments], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
