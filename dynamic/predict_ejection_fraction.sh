#!/bin/bash
#PBS -q gpu
#PBS -l walltime=96:00:00
#PBS -l mem=80g,vmem=120g
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -m n
#PBS -N echonet
#PBS -j oe
#PBS -o /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs

module load anaconda
source activate /hpf/largeprojects/ccmbio/sufkes/conda/envs/neuro
cd /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/dynamic

config_path="$1"
python /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/dynamic/echonet/utils/video.py "$config_path"

#cmd="import echonet; echonet.utils.video.run(modelname=\"r2plus1d_18\", frames=32, period=2, pretrained=True, batch_size=8, num_workers=0, run_test=True, run_train=True, load_model_weights_only=True)"
#python3 -c "${cmd}"
