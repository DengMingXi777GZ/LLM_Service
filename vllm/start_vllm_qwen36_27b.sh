#!/usr/bin/env bash
set -euo pipefail

# 限制只使用物理序号为 2 和 3 的 GPU
export CUDA_VISIBLE_DEVICES="2,3"

CONDA_SH="/mnt/data1/dmx/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="vllm_qw"
UV_VLLM_BIN="/home/dengmingxi/vllm/bin/vllm"

# 将模型路径修改为 Qwen3.6-27B 的实际存放路径
MODEL_PATH="/mnt/data1/shared/models/Qwen3.6-27B"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
# 默认服务名称修改为 Qwen3.6-27B
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-27B}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
# 新增：由于使用两张卡，需要设置张量并行度为 2
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}" 

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda init script not found: $CONDA_SH" >&2
  exit 1
fi

if [[ ! -x "$UV_VLLM_BIN" ]]; then
  echo "vLLM binary not found or not executable: $UV_VLLM_BIN" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

exec "$UV_VLLM_BIN" serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --trust-remote-code