#!/bin/bash

########################################################
# usage:
#
# 	1. exec in serial
#
#		bssh.sh "rm -r xxx.log" 0
#
# 	2. exec in background
#		
#		bssh.sh "bash xxx.sh" 1
#
########################################################

########################################################
# Fill <ip_list> with your machine ip <ifconfig eth0>
########################################################
declare -a ip_list=(
172.21.16.28
172.21.16.31
172.21.16.42
# 172.21.16.22
# 172.21.16.30
# 172.21.16.49
# 172.21.16.33
# 172.21.16.47
# 172.21.16.37
# 172.21.16.24
# 172.21.16.26
# 172.21.16.12
# 172.21.16.13
# 172.21.16.10
# 172.21.16.2
# 172.21.16.8
# 172.21.16.3
# 172.21.16.16
# 172.21.16.15
)

########################################################
# Adjust the script file path
########################################################
work_dir=$(pwd)

script_path=${1:-ray_start.sh}
model_size=${2:-}

# pip uninstall -y vllm
# pip install /cfs/haiwu/vllm-0.7.4.dev253+ga3ed5da6.d20250305-cp310-cp310-linux_x86_64.whl
# pip install nvidia-nccl-cu12==2.25.1

for i in ${ip_list[@]}
do
	echo; echo
	# cmd="cd /cfs/haiwu/deepseek; bash $script_path"
	cmd="cd /cfs/haiwu/deepseek; ray stop"
	# cmd="pip uninstall -y vllm; pip install /cfs/haiwu/vllm-0.7.4.dev253+ga3ed5da6.d20250305-cp310-cp310-linux_x86_64.whl; pip install nvidia-nccl-cu12==2.25.1"
	echo "Running [ $cmd ] on server [ $i ]"
	ssh $i "$cmd" &
done

