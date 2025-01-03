import torch
import sys
import os

# ray tune package-related functions
# import ray
# from ray import tune
# from ray.air import Checkpoint, session
# from ray.tune.schedulers import ASHAScheduler

### self defined functions
from DANN_cfDNA.raytune.ray_tune import ray_tune
from DANN_cfDNA.training.train_and_tune_1D import DANNwithTrainingTuning_1D

# default value of input_size and feature_type
feature_type_list = ["Frag","Arm","Cnv","Griffin","MCMS","Focal"]
input_size_list = [1100,950,2600,2600,200,300]

dim = "1D"
tuning_num = 50
epoch_num = 500
feature_type = "Cnv"
input_size = 2600
output_path="./Output_example/"
data_dir="./Feature_example/feature_example.pkl"
R01BTune="No"
cluster_method="kmeans"
nfold=5

best_config={'out1': 32, 'out2': 128, 'conv1': 3, 'pool1': 2, 'drop1': 0.0, 
             'conv2': 4, 'pool2': 1, 'drop2': 0.4, 'fc1': 128, 'fc2': 32, 'drop3': 0.2, 'batch_size': 128, 'num_epochs': 500, 'lambda': 0.1}

print("Default parameter set")
print(best_config)

torch.cuda.empty_cache()

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
    cluster_method = sys.argv[9]
    nfold = int(sys.argv[10])
    print(f"Getting arguments: feature type: {feature_type}, dimension: {dim}, input size: {input_size}, \
        tuning round: {tuning_num}, epoch num: {epoch_num}, output path: {output_path}, data path: {data_dir}, \
            R01BTune: {R01BTune}, clustering method: {cluster_method}, n fold: {nfold}\n")  
else:
    print(f"Not enough inputs, using default arguments: feature type: {feature_type}, input size: {input_size}, \
        tuning round: {tuning_num}, epoch num: {epoch_num}, output path: {output_path}, data path: {data_dir}, \
            R01BTune: {R01BTune}, clustering method: {cluster_method}, n fold: {nfold}\n")

## finish loading parameters from external inputs
## preset parameters
num_class = 2
num_domain = 2
gamma_r01b = 1000.0
output_path = os.path.abspath(output_path)
data_dir = os.path.abspath(data_dir)
output_path_cv = f"{output_path}_cv"
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
    
best_config["lambda"] = 1.0

#### train and tune DANN; DANNwithTrainingTuning class takes all variables in the config dictionary from ray_tune
print("***********************************   Start fittiing model with best configurations   ***********************************")
if(dim == "1D"):
    DANN_trainvalid=DANNwithTrainingTuning_1D(config=best_config, input_size=input_size,num_class=num_class,num_domain=num_domain,gamma_r01b=gamma_r01b)
# else:
#     DANN_trainvalid=DANNwithTrainingTuning(best_config, input_size=input_size,num_class=num_class)
    
DANN_trainvalid.data_loader(data_dir=data_dir,
                    input_size=input_size,
                    feature_type=feature_type,
                    R01BTuning=R01BTune_flag)

if cluster_method == "None":
    DANN_trainvalid.fit(output_path=output_path, R01BTuning_fit=R01BTune_flag)
    DANN_trainvalid.crossvalidation(num_folds=nfold,output_path=output_path)
    print("================== DANN score ======================")
    print(f"Training AUC: {DANN_trainvalid.training_auc:.4f}, R01B sensitivity: {DANN_trainvalid.testing_sens:.4f}")
    print("====================================================") 

else:
    ############################################## clustering train cancer data
    n_cluster = 2
    DANN_trainvalid.cluster_cancerdata(methods=cluster_method,encoding_size=32,n_cluster=n_cluster)

    for i in range(n_cluster):
        
        print(f"================================== Start fitting with cluster {i} ========================================")
        
        DANN_trainvalid.select_cancerdata(selected_cluster=i)
        DANN_trainvalid.weight_reset()  
        DANN_trainvalid.fit(output_path=output_path,R01BTuning_fit=R01BTune_flag)
        DANN_trainvalid.crossvalidation(num_folds=nfold,output_path=output_path)
        
        # Clear cache
        torch.cuda.empty_cache()
        # Release unused memory
        torch.cuda.memory_reserved()
        
        # output, _, _ = DANN_trainvalid(DANN_trainvalid.X_test_tensor, None, alpha = 0.1)
        # auc_traintune = roc_auc_score(DANN_trainvalid.y_test_tensor.detach().cpu(),output.detach().cpu())

        print("================== DANN score ======================")
        print(f"Training AUC: {DANN_trainvalid.training_auc:.4f}, R01B sensitivity: {DANN_trainvalid.testing_sens:.4f}")
        print("====================================================")    
        # print("================== DANN score tuned ======================")
        # print(f"Training AUC: {DANN_trainvalid.training_auc_tuned:.4f}, R01B sensitivity: {DANN_trainvalid.testing_sens_tuned:.4f}")
        # print("====================================================")  
        # print(f"----------------   Testing AUC {feature_type}: {auc_traintune}   ----------------")
        print("***********************************   Completed fitting model   ***********************************")
