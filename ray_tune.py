import torch
import torch.nn as nn
import os
import sys

from model import DANN_1D, DANN
from train_module import train_module

# ray tune
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial

def ray_tune(num_samples=200, max_num_epochs=1000, gpus_per_trial=1, 
         output_path="/mnt/binf/eric/DANN_Mercury_output/",
         data_dir="/mnt/binf/eric/Mercury_June2023_new/Feature_all_June2023_R01BMatch.csv",
         input_size=55,
         feature_type="Frag",
         dim="1D"):

    config = {
        "out1": tune.choice([2**i for i in range(4,7)]),
        "out2": tune.choice([2**i for i in range(6,9)]),
        "conv1": tune.choice([i for i in range(1,5)]),
        "pool1": tune.choice([i for i in range(1,3)]),
        "drop1": tune.choice([(i)/5 for i in range(3)]),
        
        "conv2": tune.choice([i for i in range(1,5)]),
        "pool2": tune.choice([i for i in range(1,3)]),
        "drop2": tune.choice([(i)/5 for i in range(3)]),
                                 
        "fc1": tune.choice([2**i for i in range(6,10)]),
        "fc2": tune.choice([2**i for i in range(5,9)]),
        "drop3": tune.choice([(i)/5 for i in range(5)]),
        
        "batch_size": tune.choice([2**i for i in range(7,10)]),
        "num_epochs": tune.choice([250, 500,1000,1500]),
        
        "lambda": tune.choice([0.01,0.1,0.25,0.5,1.0])
    }
    scheduler = ASHAScheduler(
        metric="testloss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2
    )
    
    ### print out processes
    if not os.path.exists(f"{output_path}/"):
        os.makedirs(f"{output_path}/")
        
    # log_output = open(f"{output_path}/{feature_type}_raytune_log_ouput.txt", "w" )
    # sys.stdout = log_output
    
    result = tune.run(
        partial(train_module,
                data_dir=data_dir,
                input_size=input_size,
                feature_type=feature_type,
                dim=dim),
        resources_per_trial={"cpu": 16, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler
    )
    
    best_trial = result.get_best_trial(metric = "testloss", mode = "min", scope = "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['testloss']}")
    print(f"Best trial final validation auc: {best_trial.last_result['testauc']}")

    if(dim == "1D"):
        best_trained_model = DANN_1D(input_size=input_size, num_class=2, num_domain=2,
                                out1=best_trial.config["out1"], out2=best_trial.config["out2"], 
                                conv1=best_trial.config["conv1"], pool1=best_trial.config["pool1"], drop1=best_trial.config["drop1"], 
                                conv2=best_trial.config["conv2"], pool2=best_trial.config["pool2"], drop2=best_trial.config["drop2"], 
                                fc1=best_trial.config["fc1"], fc2=best_trial.config["fc2"], drop3=best_trial.config["drop3"])
    else:
        best_trained_model = DANN(input_size=input_size, num_class=2, num_domain=2,
                                out1=best_trial.config["out1"], out2=best_trial.config["out2"], 
                                conv1=best_trial.config["conv1"], pool1=best_trial.config["pool1"], drop1=best_trial.config["drop1"], 
                                conv2=best_trial.config["conv2"], pool2=best_trial.config["pool2"], drop2=best_trial.config["drop2"], 
                                fc1=best_trial.config["fc1"], fc2=best_trial.config["fc2"], drop3=best_trial.config["drop3"])
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_trained_model.to(device)

    # load and save best model
    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()
    best_trained_model.load_state_dict(best_checkpoint_data["model_state_dict"])
    torch.save(best_trained_model,f"{output_path}/{feature_type}_DANN_{dim}_BestRayTune.pt")

    # log_output.close()
    
    # output config dictionary to a file
    with open(f'{output_path}/{feature_type}_config.txt','w') as config_file: 
      config_file.write(str(best_trial.config))
    config_file.close()
    
    # return config and test auc
    return best_trial.config, best_trial.last_result['testloss']
    


