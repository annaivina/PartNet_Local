#!/bin/bash
#PBS -q gpu
#PBS -N PartNet_He-kinpid
#PBS -l walltime=72:00:00
#PBS -l mem=8gb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -o output-Herwig_kinpid.log
#PBS -e error-Herwig_kinpid.log


cd /srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate common


# set the dataset dir via `DATADIR_QuarkGluon`
DATADIR='/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train_herwig/'

#For the ParticleNet 
extraopts=""
model='kinpid'
modelopts='network_config/ParticleNetConf.py'
lr="1e-2"
FEATURE_TYPE='kinpid'


python train.py \
	--data-train "${DATADIR}/train_file_*.parquet" \
	--data-test "${DATADIR}/test_file_*.parquet" \
    --data-config data_config/qg_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/PartickleNet_Herwig/${model}/net \
    --num-workers 1 --fetch-step 1 --in-memory --train-val-split 0.8889 \
    --batch-size 512 --samples-per-epoch 260000 --samples-per-epoch-val 26000 --num-epochs 50 --gpus 0 \
    --start-lr $lr --optimizer ranger --log logs/QuarkGluon__Herwig_kinpid${model}.log --predict-output pred.root \
    --tensorboard QuarkGluon__Herwig_kinpid${FEATURE_TYPE}_${model} \
    ${extraopts} "${@:3}"
