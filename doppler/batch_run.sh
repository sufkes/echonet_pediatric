#!/bin/bash

#PBS -q gpu 
#PBS -l nodes=1:ppn=1:gpus=1,mem=80g,walltime=24:00:00


source /home/sufkes/echonet
cd /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/doppler/
./run.py
