import torch
import ast
from model import DANN_1D
from load_data import load_data_1D
from sklearn.metrics import roc_auc_score

### specify feature type and input size
feature_type = "MCMS"
input_size = 200
num_class = 2
num_domain = 2
data_dir = "/mnt/binf/eric/Mercury_Aug2023_new/Feature_MGI_SeqDomain.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_path = "/mnt/binf/eric/DANN_AugResults_MGI/DANN_modelL2_0803" 

##### load best_config from text
# Specify the path to the text file
config_file = f'{output_path}/{feature_type}_config.txt'
model_file = f'{output_path}/{feature_type}_DANN_1D_BestRayTune.pt'

with open(config_file, 'r') as cf:
    config_txt = cf.read()
config_dict = ast.literal_eval(config_txt)
# Print the dictionary
print(config_dict)

model_test = torch.load(model_file)
# print(model_test)

### load data and evaluate model
_, _, _, _, X_test_tensor, y_test_tensor, _, X_all_tensor, _, _, _ = load_data_1D(data_dir, input_size, feature_type) 

X_test_tensor = X_test_tensor.to(device)
model_test.eval()
output_test, _ = model_test(X_test_tensor, alpha = 0.1)

auc_test = roc_auc_score(y_test_tensor.to('cpu').detach().numpy(), output_test.to('cpu').detach().numpy())

print(f"Test AUC: {auc_test.item():.4f}")


###### load model from train_and_tune_1D fitting; compare two models
# comparemodel_file = f"/mnt/binf/eric/DANN_JulyResults/DANN_1D_0712/Raw/{feature_type}_CNN_best.pt"
# model_compare = torch.load(comparemodel_file)
# # model_compare.eval()
# print(model_compare)

##########
# config_dict = {"out1": 16,"out2": 128,"conv1": 3,"pool1": 1,"drop1": 0,"conv2": 3,"pool2": 2,"drop2": 0.2,
#              "fc1": 256,"fc2": 64,"drop3": 0.0,"batch_size": 512,"num_epochs": 500,"lambda": 0.01}


# model_tmp = DANN_1D(input_size=input_size, num_class=2, num_domain=2,
#                                 out1=config_dict["out1"], out2=config_dict["out2"], 
#                                 conv1=config_dict["conv1"], pool1=config_dict["pool1"], drop1=config_dict["drop1"], 
#                                 conv2=config_dict["conv2"], pool2=config_dict["pool2"], drop2=config_dict["drop2"], 
#                                 fc1=config_dict["fc1"], fc2=config_dict["fc2"], drop3=config_dict["drop3"])

