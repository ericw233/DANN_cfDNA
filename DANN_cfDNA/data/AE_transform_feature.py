import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from copy import deepcopy

import matplotlib.pyplot as plt

from .load_data import load_data_1D_impute

def AE_transform(data_dir="/mnt/binf/eric/Mercury_Dec2023/Feature_all_Dec2023_Lungv2.pkl", input_size=950, feature_type="Arm", encoding_size=64, control="No"):

    ### load data; use X_r01b as AE fitting data
    data, X_train_tensor, y_train_tensor, _, _, _, _, X_all_tensor, _, _, X_r01b_tensor, _, _ = load_data_1D_impute(data_dir, input_size, feature_type) 
    
    X_train_tensor = torch.squeeze(X_train_tensor)
    X_all_tensor = torch.squeeze(X_all_tensor)
    X_r01b_tensor = torch.squeeze(X_r01b_tensor)
    
    if "Yes" in control:
        X_train_healthy = X_train_tensor[y_train_tensor == 0]
        X_train_healthy70 = X_train_healthy[:70]
        
        print("----- get first 70 KAG9 ------")
        if control == "YesHealthy":
            X_r01b_tensor = X_train_healthy70
            print("----- use only KAG9 ------")
        else:
            X_r01b_tensor = torch.cat((X_train_healthy70,X_r01b_tensor),dim=0)
        
        print(X_r01b_tensor.shape)
    

        
        
    # X_input is a NA-droped, standardized, and mean-imputed data frame, not a tensor
    train_num = int(X_r01b_tensor.shape[0] * 0.75)
    test_num = X_r01b_tensor.shape[0] - train_num
    X_r01b_train, X_r01b_test = torch.split(X_r01b_tensor,[train_num, test_num])

    # Autoencoder architecture
    class Autoencoder(nn.Module):
        def __init__(self, input_size, encoding_size):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, encoding_size)
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_size, 128),
                nn.ReLU(),
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, input_size)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    ### finish defining AE structure
    
    # Instantiate the autoencoder
    input_size = X_r01b_tensor.shape[1]
    # encoding_size = 64
    autoencoder = Autoencoder(input_size, encoding_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(autoencoder.parameters(), lr=1e-4, weight_decay=1e-5)

    # Training the autoencoder
    num_epochs = 256
    min_loss = float("inf")
    patience = 50
    
    for epoch in range(num_epochs):
        
        seed = 42 + epoch
        shuffled_indices = torch.randperm(X_r01b_train.size(0))
        X_r01b_train = X_r01b_train[shuffled_indices]
        
        outputs = autoencoder(X_r01b_train)
        loss = criterion(outputs, X_r01b_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Evaluation on test data
        with torch.no_grad():
            autoencoder.eval()
            
            test_outputs=autoencoder(X_r01b_test)
            test_loss=criterion(test_outputs, X_r01b_test)

            print(f"train loss: {loss.item():.4f}, test loss: {test_loss.item():.4f}")
            print("***********************")

            # Early stopping check
            if test_loss <= min_loss:
                min_loss = test_loss
                best_model = deepcopy(autoencoder.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered! No improvement in {patience} epochs.")
                    break
        
        
    # Encode data using the trained autoencoder
    autoencoder.train()
    autoencoder.min_loss = min_loss
    autoencoder.load_state_dict(best_model)
    
    variable_name_list = [f"AE_{feature_type}_{i}" for i in range(encoding_size)]
    encoded_all = pd.DataFrame(autoencoder.encoder(X_all_tensor).detach().numpy(), columns=variable_name_list)

    data_AE = pd.concat([data.loc[:,['SampleID','Train_Group','train','Project','R01B_label']],encoded_all],axis=1)
    
    return data_AE




