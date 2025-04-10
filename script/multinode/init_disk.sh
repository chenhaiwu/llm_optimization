#!/bin/bash

# fdisk make block
printf "n\np\n1\n\n\nw\n" | fdisk /dev/nvme0n1

mkfs.xfs -i size=512 /dev/nvme0n1p1 -f 

mkdir -p /nvme

echo UUID=$(blkid /dev/nvme0n1p1 -sUUID -ovalue) /nvme xfs defaults 0 0 >> /etc/fstab

mount -a && mount
