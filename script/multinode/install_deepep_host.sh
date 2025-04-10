#!/bin/bash

cd /cfs/haiwu/bash
yum update
yum groupinstall -y 'Development Tools'
yum install -y dkms rpm-build make

#wget https://taco-1251783334.cos.ap-shanghai.myqcloud.com/lionthu/gdrcopy-2.4.4.tar.gz
#tar -xzvf gdrcopy-2.4.4.tar.gz
cd gdrcopy-2.4.4/
make -j384
sudo make prefix=/opt/gdrcopy install

cd packages
CUDA=/usr/local/cuda ./build-rpm-packages.sh
rpm -Uvh gdrcopy-kmod-2.4.4-1dkms.el3.noarch.rpm
rpm -Uvh gdrcopy-2.4.4-1.el3.x86_64.rpm
rpm -Uvh gdrcopy-devel-2.4.4-1.el3.noarch.rpm

cd ..
./insmod.sh

