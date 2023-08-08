import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from copy import deepcopy
import sys
import os
import numpy as np

# ray tune package-related functions
import ray
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from functools import partial

### self defined functions
from model import DANN_1D
from ray_tune import ray_tune
from train_and_tune_1D import DANNwithTrainingTuning_1D
from cross_validation_1D import DANNwithCV_1D

# default value of input_size and feature_type
feature_type = "Frag"
dim = "1D"
input_size = 1200
tuning_num = 20
epoch_num = 200
output_path="/mnt/binf/eric/DANN_AugResults/DANN_1D_0807_test"
data_dir="/mnt/binf/eric/Mercury_Aug2023_new/Feature_all_Aug2023_DomainKAG9v1.csv"
R01BTune=True

best_config={'out1': 32, 'out2': 128, 'conv1': 3, 'pool1': 2, 'drop1': 0.0, 
             'conv2': 4, 'pool2': 1, 'drop2': 0.4, 'fc1': 128, 'fc2': 32, 'drop3': 0.2, 'batch_size': 128, 'num_epochs': 500, 'lambda': 0.1}
print(best_config)

### get argument values from external inputs
if len(sys.argv) >= 4:
    feature_type = sys.argv[1]
    dim = sys.argv[2]
    input_size = int(sys.argv[3])
    tuning_num = int(sys.argv[4])
    epoch_num = int(sys.argv[5])
    output_path = sys.argv[6]
    data_dir = sys.argv[7]
    R01BTune = sys.argv[8]
    print(f"Getting arguments: feature type: {feature_type}, dimension: {dim}, input size: {input_size}, \
        tuning round: {tuning_num}, epoch num: {epoch_num}, output path: {output_path}, data path: {data_dir}, R01BTune: {R01BTune}\n")  
else:
    print(f"Not enough inputs, using default arguments: feature type: {feature_type}, input size: {input_size}, \
        tuning round: {tuning_num}, epoch num: {epoch_num}, output path: {output_path}, data path: {data_dir}, R01BTune: {R01BTune}\n")

## finish loading parameters from external inputs

## preset parameters
num_class = 2
num_domain = 2
output_path_cv=f"{output_path}_cv"
config_file = f"{output_path}/{feature_type}_config.txt"
if R01BTune == "Yes":
    R01BTune_flag=True
else:
    R01BTune_flag=False


if os.path.exists(config_file):
    ##### load best_config from text
    import ast
    # Specify the path to the text file
    with open(config_file, 'r') as cf:
        config_txt = cf.read()
    config_dict = ast.literal_eval(config_txt)
    # Print the dictionary
    print(config_dict)

    best_config=config_dict
    print("***********************   Read from existing config results   ***********************************")
    print(best_config)
else:
    try:
        best_config, best_testloss=ray_tune(num_samples=tuning_num, 
                                max_num_epochs=epoch_num, 
                                gpus_per_trial=1,
                                output_path=output_path,
                                data_dir=data_dir,
                                input_size=input_size,
                                feature_type=feature_type,
                                dim=dim)
    except Exception as e:
        print("==========   Ray tune failed! An error occurred:", str(e),"   ==========")

    print("***********************   Ray tune finished   ***********************************")
    print(best_config)
    

#### train and tune DANN; DANNwithTrainingTuning class takes all variables in the config dictionary from ray_tune
print("***********************************   Start fittiing model with best configurations   ***********************************")
if(dim == "1D"):
    DANN_trainvalid=DANNwithTrainingTuning_1D(config=best_config, input_size=input_size,num_class=num_class,num_domain=num_domain)
# else:
#     DANN_trainvalid=DANNwithTrainingTuning(best_config, input_size=input_size,num_class=num_class)
    
DANN_trainvalid.data_loader(data_dir=data_dir,
                     input_size=input_size,
                     feature_type=feature_type,
                     R01BTuning=R01BTune_flag)

DANN_trainvalid.fit(output_path=output_path,
             R01BTuning_fit=R01BTune_flag)

output, _ = DANN_trainvalid(DANN_trainvalid.X_test_tensor, alpha = 0.1)
auc_traintune = roc_auc_score(DANN_trainvalid.y_test_tensor.detach().cpu(),output.detach().cpu())

print(f"----------------   Test AUC {feature_type}: {auc_traintune}   ----------------")
print("***********************************   Completed fitting model   ***********************************")

#### cv process is independent
print("***********************************   Start cross validations   ***********************************")
if(dim == "1D"):
    DANN_cv=DANNwithCV_1D(best_config,input_size=input_size,num_class=num_class,num_domain=num_domain)
# else:
#     DANN_cv=DANNwithCV(best_config, input_size=input_size,num_class=num_class)
    
DANN_cv.data_loader(data_dir=data_dir,
                    input_size=input_size,
                    feature_type=feature_type,
                    R01BTuning=R01BTune_flag)

DANN_cv.crossvalidation(output_path=output_path_cv,num_folds=5,R01BTuning_fit=R01BTune_flag)
print("***********************************   Completed cross validations   ***********************************")


