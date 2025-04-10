export VLLM_DP_MASTER_IP=${1:-localhost}
export VLLM_DP_MASTER_PORT=29500
export VLLM_DP_NODE_SIZE=${2:-1}
export VLLM_DP_NODE_RANK=${3:-0}
export PYTHONPATH=/cfs/haiwu/vllm_dp

DP=$((VLLM_DP_NODE_SIZE * 6))

CMD="VLLM_USE_V1=1 vllm serve /cfs/models/DeepSeek-V2-Lite/ \
     --trust-remote-code \
     --max_model_len 8192 \
     --max-seq-len-to-capture 8192 \
     --gpu-memory-utilization 0.8 \
     --max-num-seqs 8 \
     --tensor-parallel-size 1 \
     --data-parallel-size $DP \
     --enable-expert-parallel \
     --enforce-eager \
     --redundant-num-experts 32 \
     "
echo $CMD
eval $CMD
