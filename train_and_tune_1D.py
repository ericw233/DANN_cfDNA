import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from copy import deepcopy
import os
import inspect

from model import DANN_1D
from load_data import load_data_1D


class DANNwithTrainingTuning_1D(DANN_1D):
    def __init__(self, config, input_size, num_class, num_domain):
        model_config,_=self._match_params(config)                      # find the parameters for the original DANN class
        super(DANNwithTrainingTuning_1D, self).__init__(input_size, num_class, num_domain, **model_config)        # pass the parameters into the original DANN class
        self.batch_size=config["batch_size"]
        self.num_epochs=config["num_epochs"]
        self.loss_lambda=config["lambda"]
        
        self.criterion_task = nn.BCELoss()
        self.criterion_domain = nn.BCELoss()
        
        self.optimizer_extractor = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_task = torch.optim.Adam(self.task_classifier.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_domain = torch.optim.Adam(self.domain_classifier.parameters(), lr=1e-4, weight_decay=1e-5)
                
    def _match_params(self, config):
        model_config={}
        args = inspect.signature(DANN_1D.__init__).parameters
        model_keys = [name for name in args if name != 'self']
        # model_keys = list(self.__init__.__code__.co_varnames)[1:]

        for key, value in config.items():
            if key in model_keys:
                model_config[key] = value        
        return model_config, model_keys
    
    def data_loader(self, data_dir, input_size, feature_type, R01BTuning):
        self.input_size=input_size
        self.feature_type=feature_type
        self.R01BTuning=R01BTuning
                    
        data, X_train_tensor, y_train_tensor, d_train_tensor, X_test_tensor, y_test_tensor, _, X_all_tensor, y_all_tensor, _, train_sampleid = load_data_1D(data_dir, input_size, feature_type) 
        self.data_idonly=data[["SampleID","Train_Group"]]
        self.X_train_tensor=X_train_tensor
        self.y_train_tensor=y_train_tensor
        self.d_train_tensor=d_train_tensor
        
        self.X_test_tensor=X_test_tensor
        self.y_test_tensor=y_test_tensor
                        
        self.X_all_tensor=X_all_tensor
        self.y_all_tensor=y_all_tensor
        self.train_sampleid=train_sampleid
        
        if(self.X_train_tensor.size(0) > 0):
            print("----- data loaded -----")
            print(f"Training frame has {self.X_train_tensor.size(0)} samples")
 
        if(R01BTuning==True):
            R01B_indexes=data.loc[data["Project"].isin(["R01BMatch"])].index
            self.X_train_tensor_R01B=self.X_all_tensor[R01B_indexes]
            self.y_train_tensor_R01B=self.y_all_tensor[R01B_indexes]
        
            if(self.X_train_tensor_R01B.size(0) > 0):
                print("----- R01B data loaded -----")
                print(f"R01B train frame has {self.X_train_tensor_R01B.size(0)} samples")
            
            
    def fit(self, output_path, R01BTuning_fit):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(device)
        
        num_iterations = (self.X_train_tensor.size(0) // self.batch_size) + 1     
        patience = 100  # Number of epochs with increasing test loss before early stopping
        min_test_loss = float("inf")  # Initialize minimum test loss
        max_test_auc = float(0.0)  # Initialize maximum test auc
        best_model = None  # Initialize test model
        epochs_without_improvement = 0  # Count of consecutive epochs without improvement

        for epoch in range(self.num_epochs):
            
            self.train()
            # Mini-batch training
            seed = 42 + epoch
            shuffled_indices = torch.randperm(self.X_train_tensor.size(0))
            self.X_train_tensor = self.X_train_tensor[shuffled_indices]
            self.y_train_tensor = self.y_train_tensor[shuffled_indices]
            self.d_train_tensor = self.d_train_tensor[shuffled_indices]
            
            for batch_start in range(0, len(self.X_train_tensor), self.batch_size):
                batch_end = batch_start + self.batch_size
                ith = batch_start // self.batch_size
                p = (ith + epoch * num_iterations) / (self.num_epochs * num_iterations)
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                
                batch_X = self.X_train_tensor[batch_start:batch_end]
                batch_y = self.y_train_tensor[batch_start:batch_end]
                batch_d = self.d_train_tensor[batch_start:batch_end]

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_d = batch_d.to(device)
                
                # batch_X_source = batch_X[batch_d == 0]   # 0 being the large cluster
                # batch_y_source = batch_y[batch_d == 0]
                
                # batch_X_source = batch_X_source.to(device)
                # batch_y_source = batch_y_source.to(device)
                
                ### Forward pass
                # outputs_task, _ = self(batch_X_source, alpha)
                outputs_task, _ = self(batch_X, alpha)
                _, outputs_domain = self(batch_X, alpha)
                
                # calculate task and domain loss
                # loss_task = self.criterion_task(outputs_task, batch_y_source)
                loss_task = self.criterion_task(outputs_task, batch_y)
                loss_domain = self.criterion_domain(outputs_domain, batch_d)
                loss = loss_task + self.loss_lambda * loss_domain
                
                ##### source domain
                # Zero parameter gradients
                self.optimizer_extractor.zero_grad()
                self.optimizer_task.zero_grad()
                self.optimizer_domain.zero_grad()
                             
                # Backward and optimize
                loss.backward()
                self.optimizer_extractor.step()
                self.optimizer_task.step()
                self.optimizer_domain.step()
                
            # Print the loss after every epoch
            # train_auc = roc_auc_score(
            #     batch_y.to('cpu').detach().numpy(), outputs_task.to('cpu').detach().numpy()
            # )
            print(f"--------   Epoch: {epoch+1}/{self.num_epochs}, i: {ith}   --------")
            print(f"Train total loss: {loss.item():.4f}, Train task loss: {loss_task.item():.4f}, ")
            print("--------------------------------------")
                
            # Evaluation on test data
            with torch.no_grad():
                self.eval()
                self.X_test_tensor=self.X_test_tensor.to(device)
                self.y_test_tensor=self.y_test_tensor.to(device)
                
                test_outputs,_=self(self.X_test_tensor,alpha=0.1)
                test_outputs=test_outputs.to("cpu")

                test_loss=self.criterion_task(test_outputs, self.y_test_tensor.to("cpu"))
                test_auc = roc_auc_score(self.y_test_tensor.to("cpu"), test_outputs.to("cpu"))
                print(f"Test AUC: {test_auc.item():.4f}, Test Loss: {test_loss.item():.4f}")
                print("***********************")

                # Early stopping check
                if test_auc >= max_test_auc:
                    max_test_auc = test_auc
                    best_model = deepcopy(self.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping triggered! No improvement in {patience} epochs.")
                        break
        
        self.train()
        self.max_test_auc = max_test_auc
        self.load_state_dict(best_model)
        
        # export the best model
        if not os.path.exists(f"{output_path}/Raw/"):
            os.makedirs(f"{output_path}/Raw/")
        torch.save(self,f"{output_path}/Raw/{self.feature_type}_DANN_best.pt")
        
        # obtain scores of all samples and export
        with torch.no_grad():
            self.eval()

            self.X_all_tensor = self.X_all_tensor.to(device)
            outputs_all,_ = self(self.X_all_tensor,alpha=0.1)
            outputs_all = outputs_all.to("cpu")

            self.data_idonly['DANN_score'] = outputs_all.detach().cpu().numpy()
            self.data_idonly.to_csv(f"{output_path}/Raw/{self.feature_type}_score.csv", index=False)
                
        # fine tuning with R01BMatch data
        if(self.R01BTuning and R01BTuning_fit):
            self.train()
            optimizer_R01B = torch.optim.Adam(self.parameters(), lr=1e-6)

            # Perform forward pass and compute loss
            self.X_train_tensor_R01B = self.X_train_tensor_R01B.to(device)
            self.y_train_tensor_R01B = self.y_train_tensor_R01B.to(device)

            for epoch_toupdate in range(30):
                outputs_R01B,_ = self(self.X_train_tensor_R01B, alpha = 0.1)
                loss = self.criterion_task(outputs_R01B, self.y_train_tensor_R01B)

                # Backpropagation and parameter update
                optimizer_R01B.zero_grad()
                loss.backward()
                optimizer_R01B.step()
            
            if not os.path.exists(f"{output_path}/R01BTuned/"):
                os.makedirs(f"{output_path}/R01BTuned/")    
            torch.save(self,f"{output_path}/R01BTuned/{self.feature_type}_DANN_best_R01BTuned.pt")
            
            with torch.no_grad():
                self.eval()
                
                self.X_test_tensor=self.X_test_tensor.to(device)
                self.y_test_tensor=self.y_test_tensor.to(device)
                
                test_outputs,_=self(self.X_test_tensor,alpha=0.1)
                test_outputs=test_outputs.to("cpu")

                test_loss=self.criterion_task(test_outputs, self.y_test_tensor.to("cpu"))
                test_auc = roc_auc_score(self.y_test_tensor.to("cpu"), test_outputs.to("cpu"))
                print(f"Test AUC (tuned): {test_auc.item():.4f}, Test Loss (tuned): {test_loss.item():.4f}")
                print("*********************")
                
                ### obtain scores of all samples
                self.X_all_tensor = self.X_all_tensor.to(device)
                outputs_all_tuned,_ = self(self.X_all_tensor,alpha=0.1)
                outputs_all_tuned = outputs_all_tuned.to("cpu")

            self.data_idonly['DANN_score_tuned'] = outputs_all_tuned.detach().cpu().numpy()
            self.data_idonly.to_csv(f"{output_path}/R01BTuned/{self.feature_type}_score_R01BTuned.csv", index=False)
    
    
    def predict(self, X_predict_tensor, y_predict_tensor):
        
        X_predict_tensor = X_predict_tensor.to(self.device)
        y_predict_tensor = y_predict_tensor.to(self.device)
        with torch.no_grad():
            self.eval()
            outputs_predict,_ = self(X_predict_tensor,alpha=0.1)        
        return(outputs_predict.detach().cpu().numpy())
            
                        
                    
        
        
