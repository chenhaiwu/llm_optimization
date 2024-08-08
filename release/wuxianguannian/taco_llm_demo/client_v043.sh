#!/bin/bash

# tokenizer="/data0/models/Baichuan2-13B-Chat/"
dataset="/data0/datasets/c4_sample.jsonl"
# tokenizer="/data0/models/Qwen2-72B"
tokenizer="/data0/models/Meta-Llama-3___1-70B-Instruct"
# dataset="/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
# --lookahead-cache-config-dir . \

input_sizes=(1900 3900 31000)
for inputlen in "${input_sizes[@]}"
do
    rm -rf *.npy
    batch_sizes=(1 2 4 8 16)
    # batch_sizes=(16)
    for i in "${batch_sizes[@]}"
    do
    python benchmark_serving.py \
        --backend "taco_llm" \
        --host 127.0.0.1 \
        --port 8082 \
        --num-prompts $i \
        --tokenizer ${tokenizer} \
        --dataset ${dataset} \
        --min-prompt-len ${inputlen} \
        --max-prompt-len 131072 \
        --min-gen-len 500 \
        --trust-remote-code \
        --model ${tokenizer} \
        --max-prompts ${i} \
        --max-gen-len 550 >> llama31_70b_v51_8tp_2_lookahead_${inputlen}_500.log 2>&1
        #  --trust-remote-code >> 
    done
done


