#!/bin/bash
#PBS -q gpu
#PBS -N PartikleNet_test
#PBS -l walltime=72:00:00
#PBS -l mem=8gb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -o output-test.log
#PBS -e error-test.log


cd /srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/


# set the dataset dir via `DATADIR_QuarkGluon`
#DATADIR='/storage/agrp/annai/QURK-GLUON/datasets/pythia/QuarkGluon'
DATADIR='/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/'

#For the ParticleNet 
model='kin'
modelopts='network_config/ParticleNetConf.py'
FEATURE_TYPE='kin'


python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_NewData_epr/kin/net_epoch-19_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kin.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_NewData_epr/kinpid/net_epoch-19_state.pt' --batch-size 512 --gpus '0' --predict-output prediction_kinpid.root 


#OLD PYTHIA
python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/QG_pythia/training/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_OldData/kin/net_epoch-17_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kin_old_ep17.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/QG_pythia/training/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_OldData/kinpid/net_epoch-17_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kinpid_old_ep17.root 

#OLD HERWIG
python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/QG_herwig/training/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_OldData_Herwig/kin/net_epoch-27_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kin_old_ep27.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/QG_herwig/training/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_OldData_Herwig/kinpid/net_epoch-27_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kinpid_old_ep27.root 


python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleTransformerConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_NewData_epr_Tranformers/kin/net_epoch-19_state.pt' --batch-size 512 --gpus '0' --predict-output prediction_kin_trans.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleTransformerConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_NewData_epr_Tranformers/kinpid/net_epoch-19_state.pt' --batch-size 512 --gpus '0' --predict-output prediction_kinpid_trans.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/datasets/pythia/QuarkGluon/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleTransformerConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_OldData_Tranformers/kin/net_epoch-19_state.pt' --batch-size 512 --gpus '0' --predict-output prediction_kin_old_trans.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/datasets/pythia/QuarkGluon/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleTransformerConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_OldData_Tranformers/kinpid/net_epoch-19_state.pt' --batch-size 512 --gpus '0' --predict-output prediction_kinpid_old_trans.root 



python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/PartickleNet_newdata260k/kin/net_epoch-19_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kin.root 
python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/PartickleNet_newdata260k/kinpid/net_epoch-19_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kinpid.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleTransformerConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_Tranformers_newdata260k/kin/net_epoch-19_state.pt' --batch-size 512 --gpus '0' --predict-output prediction_kin_trans.root 
python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleTransformerConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_Tranformers_newdata260k/kinpid/net_epoch-19_state.pt' --batch-size 512 --gpus '0' --predict-output prediction_kinpid_trans.root 



#New GPS and Herwig NEW truth
python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/PartickleNet_newgpus/kin/net_epoch-21_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kin_ep21.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/PartickleNet_newgpus/kinpid/net_epoch-21_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kinpid_ep21.root 



python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train_herwig/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/PartickleNet_Herwig/kin/net_epoch-19_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kin_ep19.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train_herwig/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/PartickleNet_Herwig/kinpid/net_epoch-19_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kinpid_ep19.root 




python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleTransformerConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_Tranformers_newgpus/kin/net_epoch-19_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kin_trans.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleTransformerConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_Tranformers_newgpus/kinpid/net_epoch-19_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kinpid_trans.root 


#herwig
python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train_herwig/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleTransformerConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_Tranformers_herwig/kin/net_epoch-19_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kin_trans.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/samples_produce/Cocoa/Cocoa_Zjets/Ready_To_Train_herwig/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleTransformerConf.py --model-prefix '/srv01/agrp/annai/annai/QURK-GLUON/PartNet_Local/training/QuarkGluon_Tranformers_herwig/kinpid/net_epoch-19_state.pt' --batch-size 512 --gpus '' --predict-output prediction_kinpid_trans.root 




#The HGPFLOw inputs 

#Pythia
python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/PartNet_Local/pythia_inputs_hgpflow/test_file_*.parquet" --data-config data_config/qg_kin.yaml  --network-config network_config/ParticleNetConf.py --model-prefix '/storage/agrp/annai/QURK-GLUON/PartNet_Local/training/PartNet_hgpflow_pythia/kin/net_epoch-15_state.pt' --batch-size 512 --gpus '' --predict-output prediction_py_kin_ep15.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/PartNet_Local/pythia_inputs_hgpflow/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/storage/agrp/annai/QURK-GLUON/PartNet_Local/training/PartNet_hgpflow_pythia/kinpid/net_epoch-15_state.pt' --batch-size 512 --gpus '' --predict-output prediction_py_kinpid_ep15.root 


#Herwig
python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/PartNet_Local/herwig_inputs_hgpflow/test_file_*.parquet" --data-config data_config/qg_kin.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/storage/agrp/annai/QURK-GLUON/PartNet_Local/training/PartNet_hgpflow_herwig/kin/net_epoch-19_state.pt' --batch-size 512 --gpus '' --predict-output prediction_hw_kin_ep29.root 

python train.py --predict --data-test "/storage/agrp/annai/QURK-GLUON/PartNet_Local/herwig_inputs_hgpflow/test_file_*.parquet" --data-config data_config/qg_kinpid.yaml --network-config network_config/ParticleNetConf.py --model-prefix '/storage/agrp/annai/QURK-GLUON/PartNet_Local/training/PartNet_hgpflow_herwig/kinpid/net_epoch-19_state.pt' --batch-size 512 --gpus '' --predict-output prediction_hw_kinpid_ep29.root 



