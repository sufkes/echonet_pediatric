#!/bin/bash

#PBS -l walltime=96:00:00
#PBS -l mem=80g,vmem=80g
#PBS -l nodes=1:ppn=4
#PBS -m e
#PBS -N ENvideo
#PBS -j oe
#PBS -o /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/dynamic/output


module load anaconda
source activate /hpf/largeprojects/ccmbio/sufkes/conda/envs/neuro
#cd /hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/
cd /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/dynamic

cmd="import echonet; echonet.utils.video.run(modelname=\"r2plus1d_18\", frames=32, period=2, pretrained=True, batch_size=8, num_workers=0, run_test=True, run_train=True, load_model_weights_only=True)"
python3 -c "${cmd}"
