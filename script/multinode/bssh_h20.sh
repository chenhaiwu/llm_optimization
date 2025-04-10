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
172.21.16.6
172.21.17.58
)

cmd=${1:-date}
bg=${2:-0}

for i in ${ip_list[@]}
do
	echo; echo
	echo "Running [ $cmd ] on server [ $i ] in [ background=$bg ]"
	if [ $bg = "0" ]; then
		ssh $i "$cmd"
	else
		ssh $i "$cmd" &
	fi
done

