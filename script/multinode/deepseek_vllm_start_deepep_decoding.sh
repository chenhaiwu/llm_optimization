#!/bin/bash

export NCCL_SOCKET_IFNAME=eth0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TIMEOUT=22
export NCCL_IB_TC=160
export GLOO_SOCKET_IFNAME=eth0

export VLLM_DP_MASTER_IP=${1:-localhost}
export VLLM_DP_MASTER_PORT=29500
export VLLM_DP_NODE_SIZE=${2:-1}
export VLLM_DP_NODE_RANK=${3:-0}
#export PYTHONPATH=/workspace/DeepEP
export NVSHMEM_DIR=/opt/nvshmem
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
#export CUDA_LAUNCH_BLOCKING=1

DP=$((VLLM_DP_NODE_SIZE * 8))

#CMD="VLLM_USE_V1=1 vllm serve /cfs/models/DeepSeek-V2-Lite/ \
#--max-num-batched-tokens 128 \
#CMD="VLLM_USE_V1=1 vllm serve /cfs/models/deepseek-ai/DeepSeek-R1/ \
#CMD="VLLM_USE_V1=1 vllm serve /cfs/lionthu/models/deepseek-ai/DeepSeek-R1/ \
CMD="CUDA_LAUNCH_BLOCKING=1 VLLM_USE_V1=1 vllm serve /data0/models/deepseek-ai/DeepSeek-R1 \
     --trust-remote-code \
     --max_model_len 128 \
     --max-seq-len-to-capture 128 \
     --gpu-memory-utilization 0.80 \
     --max-num-seqs 1 \
     --tensor-parallel-size 1 \
     --data-parallel-size $DP \
     --enable-expert-parallel \
     --redundant-num-experts=0 \
     --enable-deepep-moe \
     --enforce-eager \
     --max-num-batched-tokens 128
     "

     #--enable-expert-parallel \
     # --enable-deepep-moe \
echo $CMD
eval $CMD
