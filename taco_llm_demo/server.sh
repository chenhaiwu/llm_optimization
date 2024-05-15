#!/bin/bash
# export CUDA_VISIBLE_DEVICES=
backend=${1}

# model="/workspace/suri/model/Baichuan-7B"
# model="/workspace/suri/model/Baichuan2-13B-Chat"
#model="/root/models/llama2_triton_model_repo"
# model="/model/Llama-2-70b-hf"
# model="/data/models/Qwen1.5-32B-Chat"
model="/data/models/Qwen1.5-72B-Chat"
# model="/data0/workspace/model/llama-2-70b-fp16"
# --lookahead-cache-config-dir . \
tp=8
host="127.0.0.1"
port="8080"
if [ ${backend} == "taco_llm" ];then
    python -m taco_llm.entrypoints.api_server \
        --model ${model} \
	--host ${host} \
        --port ${port} \
        --tensor-parallel-size ${tp} \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 8 \
	--max-paddings 20480 \
        --gpu-memory-utilization 0.95 \
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
