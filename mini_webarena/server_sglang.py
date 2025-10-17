import argparse
import os
import time

from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


def start_server(model_path):
    """
    First install the environment separately:
        conda create -n sglang python=3.10 -y
        conda activate sglang
        pip install "sglang[all]>=0.4.4.post3" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python"
    """
    server_process, port = launch_server_cmd(
        f"""python -m sglang.launch_server \
    --model-path {model_path} \
    --host 0.0.0.0
"""
    )
    # Wait for server to be live
    wait_for_server(f"http://localhost:{port}")
    print_highlight(f"Server launched on port: {port}")
    return server_process, port


def stop_server(server_process):
    terminate_process(server_process)
    print_highlight("Server process terminated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        default="0",
        help="Which GPU (CUDA device) to use, e.g. '0' or '1,2'. Default is '0'."
    )
    parser.add_argument(
        "--model-path",
        default="/home/zhiheng/cogito/base_models/qwen2.5-0.5b-wiki",
        help="Path to the model directory."
    )
    args = parser.parse_args()

    # Prepare environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda

    server_process = None
    # Start the server
    try:
        server_process, port = start_server(args.model_path)
        print("Server started successfully on port:", port)

        # Keep server alive until killed
        while True:
            print(f"Server is running on port: {port}")
            time.sleep(10)
    except Exception as e:
        raise RuntimeError(f"Failed to start server: {e}")
    finally:
        # Ensure we shut down the server if something goes wrong or user kills
        stop_server(server_process)


if __name__ == "__main__":
    main()
