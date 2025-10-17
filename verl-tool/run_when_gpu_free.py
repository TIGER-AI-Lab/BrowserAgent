#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run training scripts only when GPU‑0/1/2/3 are sufficiently idle.

Usage:
    python run_when_gpu_free.py
"""

import subprocess
import time
import os
import glob
import signal
from pathlib import Path

# ------------------------------ CONFIG -------------------------------- #
GPU_IDS = [0, 1, 2, 3]                 # GPUs to watch
MEM_THRESHOLD_MB = 10 * 1024           # 10 GB in MiB
CHECK_INTERVAL = 10                    # seconds between checks
SCRIPT_DIRS = [
    "/data/zhiheng/cogito/verl-tool/examples/train/wikiRL_stepExp",
    # "/data/zhiheng/cogito/verl-tool/examples/train/wikiRL_rolloutlen",
    # "/data/zhiheng/cogito/verl-tool/examples/train/wikiRL_lr",
]
# SCRIPT_NAMES = None
SCRIPT_NAMES = [ # TODO: Add script names if it is only a subset of the above

]
LOG_DIR = Path("log")                  # logs live here
CURR_FILE = Path("curr.txt")           # status file
# ---------------------------------------------------------------------- #


def write_status(msg: str) -> None:
    """Write a one‑line status into curr.txt (overwrites previous)."""
    CURR_FILE.write_text(msg + "\n")


def list_scripts() -> list[str]:
    """Return a sorted list of all .sh files under SCRIPT_DIRS."""
    paths = []
    for d in SCRIPT_DIRS:
        paths.extend(glob.glob(os.path.join(d, "*.sh")))
    return sorted(paths, key=lambda p: Path(p).name)


def gpus_are_free() -> bool:
    """
    Query nvidia‑smi and check whether every watched GPU's used memory
    is below MEM_THRESHOLD_MB.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fail fast if nvidia‑smi is not available.
        raise RuntimeError("Cannot execute nvidia-smi – is the driver installed?")

    # Parse output into integers (one line per GPU).
    used = [int(x.strip()) for x in out.strip().splitlines()]
    # Guard against machines with fewer GPUs than expected.
    watched = [used[i] for i in GPU_IDS if i < len(used)]
    return all(val < MEM_THRESHOLD_MB for val in watched)


def run_script(script_path: str) -> None:
    """Run a single .sh script and pipe both stdout & stderr to a log file."""
    script_name = Path(script_path).name
    log_path = LOG_DIR / f"{script_name}.log"
    LOG_DIR.mkdir(exist_ok=True)

    with open(log_path, "w") as log_file:
        write_status(f"Running {script_path}")
        # Use bash to execute the script.
        process = subprocess.Popen(
            ["bash", script_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        # Forward SIGINT/SIGTERM to the child so Ctrl‑C works cleanly.
        try:
            process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait()
        finally:
            write_status("Wait for GPU …")


def main():
    scripts = list_scripts()
    if not scripts:
        print("No training scripts found – nothing to do.")
        return

    write_status("Wait for GPU …")

    for s in scripts:
        # Poll until GPUs are idle enough.
        while not gpus_are_free():
            time.sleep(CHECK_INTERVAL)

        run_script(s)

    write_status("All scripts finished ✔")
    print("✅  All scripts have been executed.")


if __name__ == "__main__":
    main()
