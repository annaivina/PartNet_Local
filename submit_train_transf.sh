#!/bin/bash
#PBS -q gpu
#PBS -N PartTransformer-kinpid
#PBS -l walltime=72:00:00
#PBS -l mem=8gb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -o output-transformer_newgpus_kinpid.log
#PBS -e error-transformer_newgpus_kinpid.log


cd /srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate common


# set the dataset dir via `DATADIR_QuarkGluon`
DATADIR='/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/'

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
    --model-prefix training/QuarkGluon_Tranformers_newgpus/${model}/net \
    --num-workers 1 --fetch-step 1 --in-memory --train-val-split 0.8889 \
    --batch-size 512 --samples-per-epoch 240000 --samples-per-epoch-val 24000 --num-epochs 50 --gpus 0 \
    --start-lr $lr --optimizer ranger --log logs/QuarkGluon_Tranformers_newgpus_${model}.log --predict-output pred.root \
    --tensorboard QuarkGluon_Tranformers_newgpus_${FEATURE_TYPE}_${model} \
    ${extraopts} "${@:3}"
