#!/bin/bash

HPARAM=$1

if ! which mpstat > /dev/null; then
  sudo apt-get install sysstat -y
fi

if [[ $HPARAM == resnet_cifar_32 ]] || [[ $HPARAM == resnet_50 ]]; then
    sleep 150s
fi

sudo mpstat -P ALL 2 > ${HOME}/log.txt