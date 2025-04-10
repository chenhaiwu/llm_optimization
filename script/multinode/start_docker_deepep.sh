#!/bin/bash

docker run \
    -itd \
    -v /cfs:/cfs \
    -v /data0:/data0 \
    -v /usr/src/nvidia-535.216.01/:/usr/src/nvidia-535.216.01/ \
    --gpus all \
    --privileged --cap-add=IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --net=host \
    --ipc=host \
    --name=${1} \
    ${2}

docker exec -it ${1} bash
