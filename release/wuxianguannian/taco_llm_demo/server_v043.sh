#!/bin/bash
# export CUDA_VISIBLE_DEVICES=4,5,6,7
backend=${1}

model="/data0/models/Qwen2-72B"
# model="/data0/models/Meta-Llama-3___1-70B-Instruct"
# model="/data0/models/Llama-2-7b-chat-hf/"
# model="/data0/workspace/model/llama-2-70b-fp16"
# nsys profile --gpu-metrics-device=6  -t cuda,osrt,nvtx,cudnn,cublas -y 10 -d 90 -o /data0/haiwu/tencent1
# --use-v2-block-manager \
# --enable-prefix-caching \
# --lookahead-cache-config-dir . \

tp=4
host="127.0.0.1"
port="8082"
if [ ${backend} == "taco_llm" ];then
    python -m taco_llm.entrypoints.api_server \
        --model ${model} \
	--host ${host} \
        --port ${port} \
        --tensor-parallel-size ${tp} \
        --max-num-batched-tokens 242144 \
        --max-num-seqs 8 \
        --gpu-memory-utilization 0.95 \
        --use-v2-block-manager \
        --trust-remote-code \
        --lookahead-cache-config-dir . \
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
