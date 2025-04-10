#!bin/bash

yum update
yum install -y docker-ce docker-ce-cli containerd.io
curl -s -L https://nvidia.github.io/nvidia-docker/centos7/nvidia-docker.repo | tee /etc/yum.repos.d/nvidia-docker.repo
yum install -y nvidia-container-toolkit

cp daemon.json /etc/docker/

systemctl start docker
