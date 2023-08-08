import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import inspect

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from copy import deepcopy

from model import DANN_1D
from load_data import load_data_1D   
    
class DANNwithCV_1D(DANN_1D):
    def __init__(self, config, input_size, num_class, num_domain):
        model_config,_=self._match_params(config)
        super(DANNwithCV_1D, self).__init__(input_size, num_class, num_domain, **model_config)
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
        
        if(R01BTuning==True):
            R01B_indexes=data.loc[data["Project"].isin(["R01BMatch"])].index
            self.X_train_tensor_R01B=X_all_tensor[R01B_indexes]
            self.y_train_tensor_R01B=y_all_tensor[R01B_indexes]
    
    def weight_reset(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.reset_parameters()
        
    def crossvalidation(self,num_folds, output_path, R01BTuning_fit):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kf = KFold(n_splits=num_folds, shuffle=True)
    
        fold_scores = []  # List to store validation scores
        fold_labels = []
        fold_numbers = []
        fold_sampleid = []
        fold_scores_tuned = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(self.X_train_tensor)):
            X_train_fold, X_val_fold = self.X_train_tensor[train_index], self.X_train_tensor[val_index]
            y_train_fold, y_val_fold = self.y_train_tensor[train_index], self.y_train_tensor[val_index]
            d_train_fold = self.d_train_tensor[train_index]
            sampleid_val_fold = self.train_sampleid[val_index]
            
            ### reset the model
            self.weight_reset()
            self.to(device)
            
            num_iterations = (self.X_train_tensor.size(0) // self.batch_size) + 1          # get the iteration number
            optimizer_tuned = torch.optim.Adam(self.parameters(), lr=1e-6)
            
            patience = 100
            max_test_auc = 0.0
            best_model_cv = None
            epochs_without_improvement = 0
            
            for epoch in range(self.num_epochs):
                shuffled_indices = torch.randperm(X_train_fold.shape[0])
                X_train_fold = X_train_fold[shuffled_indices]
                y_train_fold = y_train_fold[shuffled_indices]
                d_train_fold = d_train_fold[shuffled_indices]
                ### turn to train mode
                self.train()
                
                for batch_start in range(0, X_train_fold.shape[0], self.batch_size):
                    batch_end = batch_start + self.batch_size
                    ith = batch_start // self.batch_size
                    p = (ith + epoch * num_iterations) / (self.num_epochs * num_iterations)
                    alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                    
                    batch_X = X_train_fold[batch_start:batch_end]
                    batch_y = y_train_fold[batch_start:batch_end]
                    batch_d = d_train_fold[batch_start:batch_end]
                                                                                
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    batch_d = batch_d.to(device)
                    
                    # batch_X_source = batch_X[batch_d == 0]
                    # batch_y_source = batch_y[batch_d == 0]

                    # if batch_X_source.shape[0] <= 0:
                    #     print("=======   not enough doamin 0 samples in batch_X   ========")
                    #     continue

                    # batch_X_source = batch_X_source.to(device)
                    # batch_y_source = batch_y_source.to(device)
                                       
                    # outputs_task, _ = self(batch_X_source,alpha)
                    outputs_task, _ = self(batch_X,alpha)
                    _, outputs_domain = self(batch_X,alpha)
                    
                    # loss_task = self.criterion_task(outputs_task, batch_y_source)
                    loss_task = self.criterion_task(outputs_task, batch_y)
                    loss_domain = self.criterion_domain(outputs_domain, batch_d)
                    
                    loss = loss_task + self.loss_lambda * loss_domain
                    
                    self.optimizer_extractor.zero_grad()
                    self.optimizer_task.zero_grad()
                    self.optimizer_domain.zero_grad()
                    
                    loss.backward()
                    self.optimizer_extractor.step()
                    self.optimizer_task.step()
                    self.optimizer_domain.step()
                    
                # train_auc = roc_auc_score(
                #     batch_y_source.to('cpu').detach().numpy(), outputs_task.to('cpu').detach().numpy()
                # )
                print(f"Fold: {fold+1}/{num_folds}, Epoch: {epoch+1}/{self.num_epochs}, i: {batch_start//self.batch_size}")
                print(f"Train total oss: {loss.item():.4f}, Train task oss: {loss_task.item():.4f}")
                print("-------------------------")
        
                with torch.no_grad():
                    self.eval()
                    val_outputs, _ = self(X_val_fold.to(device), alpha=0.1)
                    val_outputs = val_outputs.to("cpu")
        
                    val_loss = self.criterion_task(val_outputs.to("cpu"), y_val_fold.to("cpu"))
                    val_auc = roc_auc_score(y_val_fold.to("cpu"), val_outputs.to("cpu"))
                    print(f"Fold {fold+1}/{num_folds}, Epoch {epoch+1}/{self.num_epochs}")
                    print(f"Valid AUC: {val_auc.item():.4f}, Valid task loss: {val_loss.item():.4f}")
                    print("*************************")
        
                    if val_auc >= max_test_auc:
                        max_test_auc = val_auc
                        best_model_cv = deepcopy(self.state_dict())
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= patience:
                            print(f"Early stopping triggered for Fold {fold+1}! No improvement in {patience} epochs.")
                            break
            
            self.load_state_dict(best_model_cv)
            
            if not os.path.exists(f"{output_path}/Raw/"):
                os.makedirs(f"{output_path}/Raw/")
             
            torch.save(self, f"{output_path}/Raw/{self.feature_type}_DANN_cv_fold{fold+1}.pt")
            fold_scores.append(val_outputs.detach().cpu().numpy())  # Collect validation scores for the fold
            fold_labels.append(y_val_fold.detach().cpu().numpy())
            fold_numbers.append(np.repeat(fold+1, len(y_val_fold.detach().cpu().numpy())))
            fold_sampleid.append(sampleid_val_fold)
            
            if(R01BTuning_fit == True):
                # add model tuning with R01B
                self.train()
                for epoch_tuned in range(30):
                    self.X_train_tensor_R01B = self.X_train_tensor_R01B.to(device)
                    self.y_train_tensor_R01B = self.y_train_tensor_R01B.to(device)
                    
                    optimizer_tuned.zero_grad()
                    outputs_tuned, _ = self(self.X_train_tensor_R01B, alpha=0.1)
                    loss = self.criterion_task(outputs_tuned, self.y_train_tensor_R01B)
                    loss.backward()
                    optimizer_tuned.step()
                
                if not os.path.exists(f"{output_path}/R01BTuned/"):
                    os.makedirs(f"{output_path}/R01BTuned/")           
                torch.save(self, f"{output_path}/R01BTuned/{self.feature_type}_DANN_cv_fold{fold+1}_R01Btuned.pt")
                        
                # results of tuned model
                with torch.no_grad():
                    self.eval()
                    val_outputs, _ = self(X_val_fold.to(device), alpha=0.1)
                    val_outputs = val_outputs.to("cpu")
            
                    val_loss = self.criterion_task(val_outputs.to("cpu"), y_val_fold.to("cpu"))
                    val_auc = roc_auc_score(y_val_fold.to("cpu"), val_outputs.to("cpu"))
                    print(f"Fold {fold+1}/{num_folds}, Epoch {epoch+1}/{self.num_epochs}")
                    print(f"Valid AUC: {val_auc.item():.4f}, Valid task loss: {val_loss.item():.4f}")
                    print("************************")          
                    
                fold_scores_tuned.append(val_outputs.detach().cpu().numpy())  # Collect validation scores for the fold
                        
        all_scores = np.concatenate(fold_scores)
        all_labels = np.concatenate(fold_labels)
        all_numbers = np.concatenate(fold_numbers)
        all_sampleid = np.concatenate(fold_sampleid)
        
        # Save fold scores to CSV file
        df = pd.DataFrame({'Fold': all_numbers,
                        'Scores': all_scores,
                        'Train_Group': all_labels,
                        'SampleID': all_sampleid})
        
        if(R01BTuning_fit == True):
            all_scores_tuned = np.concatenate(fold_scores_tuned)
            df = pd.DataFrame({'Fold': all_numbers,
                            'Scores': all_scores,
                            'Scores_tuned': all_scores_tuned,
                            'Train_Group': all_labels,
                            'SampleID': all_sampleid})

        df.to_csv(f"{output_path}/{self.feature_type}_CV_score.csv", index=False)