#!/bin/bash
# Launch the PredGen API server on a GPU node
# Usage: srun --gres=gpu:1 --time=02:00:00 --pty bash slurm/run_server.sh

PORT=${1:-8765}
cd "$(dirname "$0")/../server"

echo "Starting PredGen API server on $(hostname):${PORT}..."
python -m uvicorn api_server:app --host 0.0.0.0 --port "$PORT"
