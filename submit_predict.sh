#!/bin/bash
#PBS -q gpu
#PBS -N PartikleNet_test
#PBS -l walltime=72:00:00
#PBS -l mem=8gb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -o output-test.log
#PBS -e error-test.log


cd /storage/agrp/annai/QURK-GLUON/PNet_torch/


# set the dataset dir via `DATADIR_QuarkGluon`
DATADIR='/storage/agrp/annai/QURK-GLUON/datasets/pythia/QuarkGluon'

#For the ParticleNet 
model='kin'
modelopts='network_config/ParticleNetConf.py'
FEATURE_TYPE='kin'


python train.py \
	--predict 
	--data-test "${DATADIR}/test_file_*.parquet" \
    --data-config data_config/qg_${FEATURE_TYPE}.yaml \
    --network-config $modelopts \
    --model-prefix '/storage/agrp/annai/QURK-GLUON/PNet_torch/training/QuarkGluon/kin/net_epoch-19_state.pt' \
    --batch-size 512 --gpus '0' \
    --predict-output prediction_kin.root \
