#!/bin/bash
#PBS -q gpu
#PBS -N ParticleTransformer-kinpid-old
#PBS -l walltime=72:00:00
#PBS -l mem=8gb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -o output-transformer-kinpid_old.log
#PBS -e error-transformer-kinpid_old.log


cd /srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/


# set the dataset dir via `DATADIR_QuarkGluon`
DATADIR='/storage/agrp/annai/QURK-GLUON/datasets/pythia/QuarkGluon'

#For the ParticleNet 
extraopts=""
model='kinpid'
modelopts='network_config/ParticleTransformerConf.py'
lr="1e-3"
FEATURE_TYPE='kinpid'


python train.py \
	--data-train "${DATADIR}/train_file_*.parquet" \
	--data-test "${DATADIR}/test_file_*.parquet" \
	--use-amp --optimizer-option weight_decay 0.01 \
    --data-config data_config/qg_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/QuarkGluon_OldData_Tranformers/${model}/net \
    --num-workers 1 --fetch-step 1 --in-memory --train-val-split 0.8889 \
    --batch-size 512 --samples-per-epoch 1600000 --samples-per-epoch-val 200000 --num-epochs 50 --gpus 0 \
    --start-lr $lr --optimizer ranger --log logs/QuarkGluon_OldData_Tranformers_${model}.log --predict-output pred.root \
    --tensorboard QuarkGluon_OldData_Tranformers_${FEATURE_TYPE}_${model} \
    ${extraopts} "${@:3}"
