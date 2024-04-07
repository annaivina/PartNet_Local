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
extraopts=""
model='kin'
modelopts='network_config/ParticleNetConf.py'
lr="1e-2"
FEATURE_TYPE='kin'


python train.py \
	--data-train "${DATADIR}/train_file_*.parquet" \
	--data-test "${DATADIR}/test_file_*.parquet" \
    --data-config data_config/qg_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/QuarkGluon/${model}/net \
    --num-workers 1 --fetch-step 1 --in-memory --train-val-split 0.8889 \
    --batch-size 512 --samples-per-epoch 1600000 --samples-per-epoch-val 200000 --num-epochs 20 --gpus '0' \
    --start-lr $lr --optimizer ranger --log logs/QuarkGluon_${model}.log --predict-output pred.root \
    --tensorboard QuarkGluon_${FEATURE_TYPE}_${model} \
    ${extraopts} "${@:3}"
