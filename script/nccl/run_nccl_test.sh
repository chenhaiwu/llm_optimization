#!/bin/bash

########################################################
# Fill <ip_list> with your machine ip <ifconfig eth0>
########################################################
declare -a ip_list=(
	"localhost"
)

########################################################
# bash run_nccl_test.sh <mode> <size> <num_gpus>
########################################################
# Supported mode:
#       0: allreduce
#       1: alltoall
#       2: allgather
#       3: reducescatter
#       4: broadcast
#       5: reduce
#       6: scatter
#       7: sendrecv
#e.g.
#       bash run_nccl_test.sh 0 1G 16
#
#       bash run_nccl_test.sh 1 128M 32
#
########################################################

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

HCA_NUM=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

MODE=${1:-0}
SIZE=${2:-1G}
TOTAL_NP=$(( ${#ip_list[@]} * $HCA_NUM ))
NP=${3:-$TOTAL_NP}

if [ $NP -gt $TOTAL_NP ]; then
	echo "ERROR: The specified gpu nums is larger than the total gpu nums!" && exit -1
fi
host_info=""
num_host=$(( $NP/$HCA_NUM ))
h=0
for i in "${ip_list[@]}"
do
	host_info+="$i:$HCA_NUM,"
	if [ $h -gt $num_host ]; then
		break
	fi
	((h++))
done

if [[ $MODE = "0" ]]; then
	cmd="all_reduce_perf -b 1M -e $SIZE -n 100 -f 4 -g 1"
elif [[ $MODE = "1" ]]; then
	cmd="alltoall_perf -b 1M -e $SIZE -n 100 -f 4 -g 1"
elif [[ $MODE = "2" ]]; then
	cmd="all_gather_perf -b 1M -e $SIZE -n 100 -f 4 -g 1"
elif [[ $MODE = "3" ]]; then
	cmd="reduce_scatter_perf -b 1M -e $SIZE -n 100 -f 4 -g 1"
elif [[ $MODE = "4" ]]; then
	cmd="broadcast_perf -b 1M -e $SIZE -n 100 -f 4 -g 1"
elif [[ $MODE = "5" ]]; then
	cmd="reduce_perf -b 1M -e $SIZE -n 100 -f 4 -g 1"
elif [[ $MODE = "6" ]]; then
	cmd="scatter_perf -b 1M -e $SIZE -n 100 -f 4 -g 1"
elif [[ $MODE = "7" ]]; then
	cmd="sendrecv_perf -b 1M -e $SIZE -n 100 -f 4 -g 1"
else
	echo "ERROR: Unsupported running mode and exit" && exit -1
fi

mpi_cmd="mpirun -np $NP \
	 -H ${host_info::-1} \
	 --allow-run-as-root -bind-to none -map-by slot \
	 -x NCCL_DEBUG=INFO \
	 -x NCCL_IB_GID_INDEX=3 \
	 -x NCCL_IB_DISABLE=0 \
	 -x NCCL_SOCKET_IFNAME=eth0 \
	 -x NCCL_NET_GDR_LEVEL=2 \
	 -x NCCL_IB_QPS_PER_CONNECTION=4 \
	 -x NCCL_IB_TC=160 \
	 -x NCCL_IB_TIMEOUT=22 \
	 -x NCCL_PXN_DISABLE=0 \
	 -x NCCL_MIN_CTAS=4 \
	 -x LD_LIBRARY_PATH -x PATH \
	 -mca coll_hcoll_enable 0 -mca pml ob1 -mca btl_tcp_if_include eth0 -mca btl ^openib \
	 $cmd "

echo $mpi_cmd
eval $mpi_cmd
