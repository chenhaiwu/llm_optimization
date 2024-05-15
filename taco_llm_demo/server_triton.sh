#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
backend=${1}

# model="/workspace/suri/model/baichuan-7b"
model="/workspace/suri/coding/tensorrtllm_backend/triton_model_repo"

tp=1
host="127.0.0.1"
port="8081"

if [ ${backend} == "taco_llm" ];then
    python -m taco_llm.entrypoints.api_server \
        --model ${model} \
	--host ${host} \
        --port ${port} \
        --tensor-parallel-size ${tp} \
        --max-num-batched-tokens 20480 \
        --max-num-seqs 64 \
	--max-paddings 20480 \
        --gpu-memory-utilization 0.95 \
	--trust-remote-code \
        --disable-log-requests
elif [ ${backend} == "trt_llm" ];then
    python3 launch_triton_server.py \
        --world_size=${tp} \
	--http_port=${port} \
	--model_repo=${model}
else
    echo "Not supported backend: ${backend}"
    exit -1
fi
