#!/bin/bash

LLAMA2_DATASET_URL="https://taco-1251783334.cos.ap-shanghai.myqcloud.com/dataset/oscar/oscar-llama2.tgz"
if [ -f "/data/datasets/oscar-llama2/oscar_text_document.bin" ]; then
        echo "The oscar dataset has been already downloaded for training."
else
        mkdir -p /data/datasets/
        pushd /data/datasets/
        curl $LLAMA2_DATASET_URL -o oscar-llama2.tgz
        tar xf oscar-llama2.tgz && rm -f oscar-llama2.tgz
        popd
fi

GPT_DATASET_URL="https://taco-1251783334.cos.ap-shanghai.myqcloud.com/dataset/gpt_data/gpt-data.tgz"
if [ -f "/data/datasets/gpt-data/my-gpt2_text_document.bin" ]; then
        echo "The gpt dataset has been already downloaded for training."
else
        mkdir -p /data/datasets/
        pushd /data/datasets/
        curl $GPT_DATASET_URL -o gpt-data.tgz
        tar xf gpt-data.tgz && rm -f gpt-data.tgz
        popd
fi

MIXTRAL_DATASET_URL="https://taco-1251783334.cos.ap-shanghai.myqcloud.com/dataset/alpaca/alpaca-mixtral.tgz"
if [ -f "/data/datasets/alpaca-mixtral/alpaca_packed_input_ids_document.bin" ]; then
        echo "The mixtral dataset has been downloaded for training."
else
        mkdir -p /data/datasets/
        pushd /data/datasets/
        curl $MIXTRAL_DATASET_URL -o alpaca-mixtral.tgz
        tar xf alpaca-mixtral.tgz && rm -f alpaca-mixtral.tgz
        popd
fi
