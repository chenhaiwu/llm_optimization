#!/bin/bash

# tokenizer="/workspace/suri/model/Baichuan-7B"
# tokenizer="/workspace/suri/model/llama-2-13b-Chat-fp16"
# tokenizer="/workspace/suri/model/Baichuan2-13B-Chat"
tokenizer="/model/Llama-2-70b-hf"
# tokenizer="/workspace/suri/model/llama-2-70b-fp16"
dataset="/model/ShareGPT_V3_unfiltered_cleaned_split.json"
# /data/suri/coding/taco_llm_demo/ShareGPT_V3_unfiltered_cleaned_split.json
# prompts_array=(16)
prompts_array=(1 2 4 6 8 10 12 16)
for prompts in "${prompts_array[@]}"
do
    python benchmark_serving.py \
        --backend ${1} \
        --host 127.0.0.1 \
        --port 8081 \
        --num-prompts ${prompts} \
        --tokenizer ${tokenizer} \
        --dataset ${dataset} \
        --min-prompt-len 1500 \
        --max-prompt-len 1600 \
        --min-gen-len 200 \
        --max-gen-len 200 \
        --request-rate inf \
        --trust-remote-code > ${prompts}_base_1k5_1k6.log 2>&1 
        # > ${prompts}_lookahead_warm2_1k5_1k6.log 2>&1 
done