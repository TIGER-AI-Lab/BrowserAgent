"""
utils_sglang_local.py
~~~~~~~~~~~~~~~~~~~~~

Spin‑up and manage an **sglang** inference server from Python,
without hard‑coding a port and without leaving orphaned processes.

Usage
-----
>>> from utils_sglang_local import call_sglang_local
>>> answer = call_sglang_local("Hello, world!")
>>> print(answer)

Run this file directly to perform a quick self‑check:
$ python utils_sglang_local.py --prompt "2+2=?"
"""

from __future__ import annotations

import atexit
import os
import signal
import time
from types import SimpleNamespace
from typing import Optional

import requests
from sglang.utils import (
    wait_for_server,
    print_highlight,
    terminate_process,
    launch_server_cmd,
)

# --------------------------------------------------------------------------- #
# Helpers to start / stop an sglang server
# --------------------------------------------------------------------------- #
import socket
import subprocess
from contextlib import closing
from sglang.utils import wait_for_server, print_highlight   # unchanged

def _find_free_port() -> int:
    """Ask the OS for a free TCP port and return it."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def start_server(
    model_path: str,
    python_bin: str = "python",
    host: str = "localhost",
):
    """
    Launch an `sglang.launch_server` process with the *given* Python interpreter
    and pipe all server logs to **sglang_server.log** (truncating each time).

    Returns
    -------
    (subprocess.Popen, int)
        The process handle and the port we chose.
    """
    # Choose an available port ourselves
    port = _find_free_port()

    # Command split into argv list (no shell redirection tokens!)
    cmd = [
        python_bin, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", host,
        "--port", str(port),
    ]

    # Open (and truncate) log file *before* starting the process
    log_file = open("sglang_server.log", "w")

    # Start the server; stdout/stderr both → log file
    server_process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Block until the HTTP endpoint is ready
    wait_for_server(f"http://localhost:{port}")

    print_highlight(
        f"[sglang‑local] Server launched on port {port} using {python_bin} "
        "(logs ➜ sglang_server.log)"
    )

    return server_process, port


def stop_server(proc):
    """Terminate the subprocess gracefully (idempotent)."""
    if proc is not None:
        terminate_process(proc)
        print_highlight("[sglang‑local] Server process terminated.")


# --------------------------------------------------------------------------- #
# Stateful client wrapper
# --------------------------------------------------------------------------- #
class SGLangLocalClient:
    """
    Keep exactly **one** server process alive and route prompts to it.

    Clean‑up hooks ensure the subprocess dies on normal interpreter exit
    and on SIGINT / SIGTERM.
    """

    def __init__(
        self,
        model_path: str,
        cuda: str = "0",
        python_bin: str = "python",
        temperature: float = 0.0,
        max_new_tokens: int = 4096,
    ):
        # Generation config
        self.gen_cfg = SimpleNamespace(
            provider="huggingface",
            model="Qwen/Qwen2.5-7B-Instruct",
            gen_config={
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        )

        # Select GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda

        # Launch server
        self._server_process, self._port = start_server(
            model_path=model_path,
            python_bin=python_bin,
        )

        # Register clean‑up hooks
        atexit.register(self._safe_kill)
        signal.signal(signal.SIGINT, self._signal_handler)   # Ctrl‑C
        signal.signal(signal.SIGTERM, self._signal_handler)  # `kill`

    # ----------------------------- public API ----------------------------- #
    def __call__(self, prompt: str) -> str:
        """Send one prompt, return assistant response."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be str")

        url = f"http://localhost:{self._port}/v1/chat/completions"
        data = {
            "model": self.gen_cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.gen_cfg.gen_config["temperature"],
            "max_tokens": self.gen_cfg.gen_config["max_new_tokens"],
        }

        retries = 0
        while retries < 20:
            try:
                r = requests.post(url, json=data, timeout=120)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                print(f"[sglang‑local] Error: {exc} — retry {retries+1}/20")
                retries += 1
                time.sleep(5)

        raise RuntimeError("sglang request failed after 10 retries")

    def stop(self):
        """Manually shut down the subprocess (optional)."""
        self._safe_kill()

    # ---------------------------- context mgr ----------------------------- #
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._safe_kill()

    # --------------------------- internal utils --------------------------- #
    def _safe_kill(self, *_):
        """Terminate the server once; subsequent calls are no‑ops."""
        if getattr(self, "_server_process", None):
            stop_server(self._server_process)
            self._server_process = None

    def _signal_handler(self, signum, _frame):
        print_highlight(f"[sglang‑local] Caught signal {signum}; shutting down.")
        self._safe_kill()
        # Propagate the signal so shell exit codes remain correct
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


# --------------------------------------------------------------------------- #
# Functional singleton wrapper
# --------------------------------------------------------------------------- #
_client_singleton: Optional[SGLangLocalClient] = None

def init_sglang_local(
    model_path: str = "Qwen/Qwen3-4B",
    cuda: str = "0",
    python_bin: str = "/minimax-dialogue/ruobai/cogito/verl-tool/mini_webarena/.venv/bin/python"
):
    """
    Convenience wrapper to create a singleton client.

    Parameters mirror those of `SGLangLocalClient` except that *model_path* is
    passed to the constructor.
    """
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = SGLangLocalClient(
            model_path=model_path,
            cuda=cuda,
            python_bin=python_bin,
        )

def call_sglang_local(
    prompt: str,
):
    """
    Convenience wrapper that keeps a single global client alive.

    Parameters mirror those of `SGLangLocalClient` except that *prompt* is first.
    """
    global _client_singleton
    return _client_singleton(prompt)


# --------------------------------------------------------------------------- #
# Quick self‑test when executed directly
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick sglang‑local test.")
    parser.add_argument("--prompt", default="Hello, sglang! What is the answer of 1+1?", help="Prompt text")
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-4B",
        help="Model directory",
    )
    parser.add_argument("--cuda", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument(
        "--python-bin",
        default="/minimax-dialogue/ruobai/cogito/verl-tool/mini_webarena/.venv/bin/python",
        help="Absolute path to python interpreter in the venv running sglang",
    )
    args = parser.parse_args()

    try:
        # Initialize the singleton client
        init_sglang_local(
            model_path=args.model_path,
            cuda=args.cuda,
            python_bin=args.python_bin,
        )
        response = call_sglang_local(
            prompt=args.prompt,
            # model_path=args.model_path,
            # cuda=args.cuda,
            # python_bin=args.python_bin,
        )
        print(f"\n--- Prompt ---\n{args.prompt}\n")
        print_highlight("\n--- Model response ---")
        print(response)
    finally:
        # Ensure we do not leave the server running after the test
        if _client_singleton is not None:
            _client_singleton.stop()
