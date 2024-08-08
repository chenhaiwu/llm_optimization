#!/bin/bash

# tokenizer="/data0/models/Baichuan2-13B-Chat/"
dataset="/data0/datasets/c4_sample.jsonl"
tokenizer="/data0/models/Qwen2-72B"
# tokenizer="/data0/models/Meta-Llama-3___1-70B-Instruct"
# dataset="/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
# --lookahead-cache-config-dir . \

input_sizes=(4000 15000)
for inputlen in "${input_sizes[@]}"
do
    rm -rf *.npy
    batch_sizes=(4 8)
    # batch_sizes=(1)
    for i in "${batch_sizes[@]}"
    do
    python benchmark_speed_taco.py \
        --host 127.0.0.1 \
        --port 8082 \
        --num-prompts $i \
        --tokenizer ${tokenizer} \
        --dataset ${dataset} \
        --min-prompt-len ${inputlen} \
        --max-prompt-len 131072 \
        --min-gen-len 400 \
        --parallel ${i} \
        --num-prompts $(($i*4)) \
        --max-gen-len 450 >> qwen2_72b_wx_v51_8tp_2_lookahead_${inputlen}_400_parallel_4.log 2>&1
        #  --trust-remote-code >> 
    done
done


