#!/bin/bash

cd /workspace/DeepEP
git checkout .
rm -r deep_ep_cpp.cpython-310-x86_64-linux-gnu.so
git apply /cfs/haiwu/deepep_h20.patch

export NVSHMEM_DIR=/opt/nvshmem
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"

NVSHMEM_DIR=/opt/nvshmem python setup.py build
ln -s build/lib.linux-x86_64-cpython-310/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so

