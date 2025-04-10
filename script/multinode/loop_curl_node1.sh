#!/bin/bash

for((i=0;i<8;i++)) do bash test_curl.sh >> logs_deepep_2/log_node1_$i.log & done
