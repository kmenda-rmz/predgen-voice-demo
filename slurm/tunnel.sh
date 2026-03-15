#!/bin/bash
# SSH tunnel for accessing GPU node from MacBook
# Usage: ./tunnel.sh <gpu-node-hostname> [port]
# e.g., ./tunnel.sh gpu-node-042 8765

GPU_NODE=${1:?"Usage: ./tunnel.sh <gpu-node-hostname> [port]"}
PORT=${2:-8765}
LOGIN_NODE="user30@184.34.82.180"
SSH_KEY="$HOME/.ssh/id_hackathon"

echo "Setting up SSH tunnel: localhost:${PORT} → ${GPU_NODE}:${PORT} via ${LOGIN_NODE}"
ssh -i "$SSH_KEY" -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -N -L "${PORT}:${GPU_NODE}:${PORT}" "$LOGIN_NODE"
