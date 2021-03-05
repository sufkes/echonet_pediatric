#!/bin/bash

#PBS -l walltime=96:00:00
#PBS -l mem=80g,vmem=80g
#PBS -l nodes=1:ppn=4
#PBS -m e
#PBS -N ENsegment
#PBS -j oe
#PBS -o /hpf/largeprojects/ccmbio/sufkes/echonet/dynamic


module load anaconda
source activate /hpf/largeprojects/ccmbio/sufkes/conda/envs/neuro
#cd /hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/
cd /hpf/largeprojects/ccmbio/sufkes/echonet/dynamic

cmd="import echonet; echonet.utils.segmentation.run(modelname=\"deeplabv3_resnet50\", save_segmentation=True, pretrained=True)"
python3 -c "${cmd}"
