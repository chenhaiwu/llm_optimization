#!bin/bash

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export GLOO_SOCKET_IFNAME=eth0

export PYTHONPATH=/workspace/DeepEP

export MASTER_ADDR=${1:-localhost}
export MASTER_PORT=8000
export WORLD_SIZE=${2:-1}
export RANK=${3:-0}

python /workspace/DeepEP/tests/test_internode.py
