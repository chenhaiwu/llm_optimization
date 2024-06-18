#! /bin/bash

####################################################
#
# usage:
#      bash start.sh <model_size> <master_addr> <node_num> <rank>
#
# supported model size: {7, 13, 70}
#
####################################################

# env var
export CUDA_DEVICE_MAX_CONNECTIONS=1

# nccl settings
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22

export GLOO_SOCKET_IFNAME=eth0

# data settings
CHECKPOINT_PATH="./output"
DATA_PATH="/data/datasets/oscar-llama2/oscar_text_document"
TOKENIZER_PATH="/data/datasets/oscar-llama2/tokenizer.model"

# model settings
SEQ_LEN=16
MAX_SEQ_LEN=16
MODEL_SIZE=${1:-8}
if [ $MODEL_SIZE == "8" ]; then
        NUM_LAYERS=8
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=14336
        NUM_ATTN_HEADS=32
        MICRO_BATCH_SIZE=1
        TP=8
        PP=1
        CP=1
        MICRO_BATCH_NUM=32
        NUM_KV_HEADS=8
        GQA_OPTS=" --group-query-attention --num-query-groups $NUM_KV_HEADS "
elif [ $MODEL_SIZE == "7" ]; then
        NUM_LAYERS=32
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=11008
        NUM_ATTN_HEADS=32
        MICRO_BATCH_SIZE=1
        TP=1
        PP=1
        CP=4
        MICRO_BATCH_NUM=1
        #NUM_KV_HEADS=8
        #GQA_OPTS=" --group-query-attention --num-query-groups $NUM_KV_HEADS "
elif [ $MODEL_SIZE == "13" ]; then
        NUM_LAYERS=40
        HIDDEN_SIZE=5120
        FFN_HIDDEN_SIZE=13824
        NUM_ATTN_HEADS=40
        MICRO_BATCH_SIZE=1
        TP=1
        PP=4
        MICRO_BATCH_NUM=128
        GQA_OPTS=""
elif [ $MODEL_SIZE == "70" ]; then
        NUM_LAYERS=80
        HIDDEN_SIZE=8192
        FFN_HIDDEN_SIZE=28672
        NUM_ATTN_HEADS=32
        MICRO_BATCH_SIZE=2
        TP=8
        PP=4
        MICRO_BATCH_NUM=128
        NUM_KV_HEADS=8
        GQA_OPTS=" --group-query-attention --num-query-groups $NUM_KV_HEADS "
else
        echo "ERROR: Please supplement new model configuration to test!"
        exit -1
fi

#fp8 settings
ENABLE_FP8=false
if [ $ENABLE_FP8 == "true" ]; then
        FP8_OPTS="--transformer-impl transformer_engine --fp8-format hybrid "
        DT="fp8"
else
        FP8_OPTS="--transformer-impl transformer_engine"
        DT="bf16"
fi

# node settings
MASTER_ADDR=${2:-localhost}
MASTER_PORT=6000
NNODES=${3:-1}
NODE_RANK=${4:-0}
GPUS_PER_NODE=4
WORLD_SIZE=$(( $GPUS_PER_NODE * $NNODES ))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

DP=$(( $WORLD_SIZE / $TP / $PP / $CP))
GLOBAL_BATCH_SIZE=$(( $DP * $MICRO_BATCH_SIZE * $MICRO_BATCH_NUM ))

echo "DP" 
echo $DP

CMD="torchrun $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --use-mcore-models \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --tensor-model-parallel-size $TP \
        --pipeline-model-parallel-size $PP \
	--sequence-parallel \
        --context-parallel-size $CP\
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --ffn-hidden-size $FFN_HIDDEN_SIZE \
        --num-attention-heads $NUM_ATTN_HEADS \
        $GQA_OPTS \
        --hidden-dropout 0.0 \
        --attention-dropout 0 \
        --swiglu \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --position-embedding-type rope \
        --normalization RMSNorm \
        --train-iters 1 \
        --lr-decay-iters 320000 \
        --load $CHECKPOINT_PATH \
        --save $CHECKPOINT_PATH \
        --no-load-optim \
        --no-load-rng \
        --data-path $DATA_PATH \
        --tokenizer-type Llama2Tokenizer \
        --tokenizer-model ${TOKENIZER_PATH} \
        --split 8,1,1 \
        --distributed-backend nccl \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction .01 \
        --log-interval 1 \
        --log-throughput \
        --save-interval 10000 \
        --eval-interval 10000 \
        --exit-interval 10000 \
        --eval-iters 1 \
        --use-flash-attn \
        --use-distributed-optimizer \
        --bf16 \
        $FP8_OPTS \
        "

echo ${CMD} 2>&1 | tee megatron_llama2-${MODEL_SIZE}B_tp${TP}_pp${PP}_dp${DP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${DT}.log
eval ${CMD} 2>&1 | tee -a megatron_llama2-${MODEL_SIZE}B_tp${TP}_pp${PP}_dp${DP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${DT}.log

