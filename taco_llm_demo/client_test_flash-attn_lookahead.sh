#!/bin/bash

# tokenizer="/data0/workspace/model/llama-2-70b-fp16/"
# tokenizer="/model/Llama-2-70b-hf"
tokenizer="/data/models/Qwen1.5-32B-Chat"
# dataset="/workspace/taco-llm/demo/llama/13b/sample_input_file.txt"
# dataset="/model/ShareGPT_V3_unfiltered_cleaned_split.json"
dataset="/data/datasets/c4_sample.jsonl"

rm -rf *.npy

batch_sizes=(1 2 4 8 16 32)
# batch_sizes=(1)
for i in "${batch_sizes[@]}"
do
python benchmark_serving.py \
    --backend "taco_llm" \
    --host 127.0.0.1 \
    --port 8080 \
    --num-prompts $i \
    --tokenizer ${tokenizer} \
    --dataset ${dataset} \
    --min-prompt-len 480 \
    --max-prompt-len 32768 \
    --min-gen-len 10 \
    --max-gen-len 200 >> qwen2-34b_2tp_500_400_flash.log 2>&1
    #  --trust-remote-code >> 
done
