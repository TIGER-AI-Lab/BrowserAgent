#!/bin/bash

# VLLM model deployment script (supports external model path input)
# Usage: bash deploy_vllm.sh /path/to/your/model

# Argument check
if [ -z "$1" ]; then
  echo "❌ Error: Please provide the model path as the first argument."
  echo "Usage: bash $0 /path/to/your/model"
  exit 1
fi

# Retrieve the model path provided by the user
MODEL_PATH="$1"

# Server configuration (based on OpenAI client settings from run_model.py)
HOST="localhost"
PORT=5001

# VLLM parameter configuration
TENSOR_PARALLEL_SIZE=1        # Number of GPUs used for tensor parallelism
MAX_MODEL_LEN=32768           # Maximum sequence length
GPU_MEMORY_UTILIZATION=0.9    # GPU memory utilization; can be increased when using 8 GPUs

# Set GPU environment variables
export CUDA_VISIBLE_DEVICES=0           # Specify which GPU(s) to use
export NCCL_P2P_DISABLE=1               # Disable P2P communication if necessary
export NCCL_IB_DISABLE=1                # Disable InfiniBand if you encounter related issues

echo "🚀 Starting the VLLM server..."
echo "📁 Model path: $MODEL_PATH"
echo "🌐 Server address: http://$HOST:$PORT"
echo "🎮 GPUs in use: $CUDA_VISIBLE_DEVICES"
echo "🧠 Tensor parallel size: $TENSOR_PARALLEL_SIZE"

# Launch the VLLM OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --served-model-name "qwen2.5-7b" \
    --trust-remote-code \
    --disable-log-requests \
    --api-key "sk-proj-1234567890"

echo "✅ VLLM server has started successfully!"
