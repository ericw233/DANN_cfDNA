#!/mnt/binf/eric/anaconda3/envs/Py38/bin/python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from copy import deepcopy
import os
import inspect
import matplotlib.pyplot as plt

from ..model.model_3layer import DANN_1D
from ..data.load_data import load_data_1D_impute
from ..utils.AE_clustering import AE_clustering
from ..utils.find_threshold import find_threshold, find_sensitivity

class DANNwithTrainingTuning_1D(DANN_1D):
    def __init__(self, config, input_size, num_class, num_domain, gamma_r01b):
        model_config,_=self._match_params(config)                      # find the parameters for the original DANN class
        super(DANNwithTrainingTuning_1D, self).__init__(input_size, num_class, num_domain, **model_config)        # pass the parameters into the original DANN class
        self.batch_size=config["batch_size"]
        self.num_epochs=config["num_epochs"]
        self.loss_lambda=config["lambda"]
        self.gamma_r01b=gamma_r01b
        
        self.criterion_task = nn.BCELoss()
        self.criterion_domain = nn.BCELoss()
        self.criterion_r01b = nn.L1Loss()
        self.criterion_r01b_ranking = nn.MarginRankingLoss()
        
        self.optimizer_extractor = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_task = torch.optim.Adam(self.task_classifier.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_domain = torch.optim.Adam(self.domain_classifier.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_r01b = torch.optim.Adam(self.r01b_classifier.parameters(), lr=1e-4, weight_decay=1e-5)
                
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
        self.methods="NoClustering"
        
        self.selected_cluster=0 # set default value of selected cluster
                    
        data, X_train_tensor, y_train_tensor, d_train_tensor, X_test_tensor, y_test_tensor, _, X_all_tensor, y_all_tensor, d_all_tensor, X_r01b_tensor, y_r01b_tensor, train_sampleid = load_data_1D_impute(data_dir, input_size, feature_type) 
        self.data_idonly=data[["SampleID","Train_Group","train","Project","Domain","R01B_label"]]
        self.data_idonly['Train_Group'] = self.data_idonly['Train_Group'].replace({'Healthy':0,'Cancer':1})
        
        ### X_train_tensor_original would be kept unchanged
        self.X_train_tensor_original=X_train_tensor
        self.y_train_tensor_original=y_train_tensor
        self.d_train_tensor_original=d_train_tensor
        
        self.X_test_tensor=X_test_tensor
        self.y_test_tensor=y_test_tensor
                        
        self.X_all_tensor=X_all_tensor
        self.y_all_tensor=y_all_tensor
        self.d_all_tensor=d_all_tensor
        self.sampleid_train=train_sampleid
        
        # this r01b tensor contains 70 samples for r01b classifier
        self.X_r01b_tensor=X_r01b_tensor
        self.y_r01b_tensor=y_r01b_tensor
                
        ### X_train_tensor is set for model fitting, may be modified by following modules
        self.X_train_tensor=X_train_tensor
        self.y_train_tensor=y_train_tensor
        self.d_train_tensor=d_train_tensor 
                
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
    
    # to modify the cancer frame - select one of the clusters        
    def cluster_cancerdata(self, methods = "kmeans", encoding_size = 256, n_cluster = 4):
        
        self.n_cluster = n_cluster
        self.methods = methods
        # prepare cancer and healthy 
        X_train = pd.DataFrame(self.X_train_tensor_original.view(self.X_train_tensor_original.shape[0],-1).numpy())
        y_train = pd.DataFrame(self.y_train_tensor_original.numpy())
        
        X_train_cancer = X_train.loc[y_train.iloc[:,0] == 1]
        sampleid_train_cancer = self.sampleid_train[y_train.values[:,0] == 1]
        # sampleid_train_healthy = sampleid_train.loc[sampleid_train['Train_Group'] == 0,"SampleID"]
        
        # this data frame contains SampleID, Cluster, and X_train_cancer contents
        self.sampleid_cluster_df = AE_clustering(X_train_cancer, sampleid_train_cancer, methods = "kmeans", encoding_size = encoding_size, n_cluster = n_cluster)
        print(f"The size of each cluster is {pd.crosstab(self.sampleid_cluster_df['Cluster'], 1)}")
    
    def select_cancerdata(self,selected_cluster = 0):
        
        # assign selected cluster
        self.selected_cluster=selected_cluster
        
        data_all_df = pd.concat([self.data_idonly, pd.DataFrame(self.X_all_tensor.view(self.X_all_tensor.shape[0],-1).cpu().numpy())], axis=1)
        
        # ### prepare X_train_healthy
        data_trainhealthy_df = data_all_df.loc[(data_all_df['Train_Group'] == 0) & (data_all_df['train'] == 'training')]
        data_trainhealthy_df.loc[:,'Cluster'] = -1
        
        data_traincancer_df = pd.merge(data_all_df, self.sampleid_cluster_df.loc[:,['SampleID','Cluster']], on="SampleID", how="inner")
        data_traincancer_df['Cluster'] = data_traincancer_df['Cluster'].fillna(-1)
        
        data_traincancer_selected = data_traincancer_df.loc[(data_traincancer_df['Cluster'] == selected_cluster)]
        
        data_train_selected = pd.concat([data_traincancer_selected, data_trainhealthy_df])
              
        # data_train_selected.to_csv(f"/mnt/binf/eric/DANN_Jan2024Results_new/DANN_0208v2/data_train_check.csv")
        print("===============------------------------===============")
        print(pd.crosstab(data_train_selected['Train_Group'],data_train_selected['train']))      
                

        X_train = data_train_selected.drop(columns = (self.data_idonly.columns.tolist() + ['Cluster']))
        y_train = data_train_selected.loc[:,"Train_Group"]
        d_train = data_train_selected.loc[:,"Domain"]
        
        print(f"------------------------ shape of X_train: {X_train.shape} ---------------------------------")
        print(f"------------------------ shape of y_train: {y_train.shape} ---------------------------------")
        
        # X_train = data_all_df.drop(columns = (self.data_idonly.columns.tolist()))
        # y_train = data_all_df.loc[:,"Train_Group"]
        # d_train = data_all_df.loc[:,"Domain"]
        
        self.X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
        self.y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).resize_(X_train.shape[0])
        self.d_train_tensor = torch.tensor(d_train.values, dtype=torch.float32).resize_(X_train.shape[0])
        self.sampleid_train = data_train_selected["SampleID"].values
     
    # to reset the network
    def weight_reset(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.reset_parameters()
    
            
    def fit(self, output_path, R01BTuning_fit):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(device)
        
        num_iterations = (self.X_train_tensor.size(0) // self.batch_size) + 1     
        self.patience = 50  # Number of epochs with increasing test loss before early stopping
        min_test_loss = float("inf")  # Initialize minimum test loss
        max_test_auc = float(0.0)  # Initialize maximum test auc
        best_model = None  # Initialize test model
        epochs_without_improvement = 0  # Count of consecutive epochs without improvement

        # self.X_r01b_tensor=self.X_r01b_tensor.to(device)
        # self.y_r01b_tensor=self.y_r01b_tensor.to(device)

        indice_r01b_cancer = torch.where(self.y_r01b_tensor == 1)
        X_r01b_cancer_tensor = self.X_r01b_tensor[indice_r01b_cancer]
        y_r01b_cancer_tensor = self.y_r01b_tensor[indice_r01b_cancer]
            
        
        train_losses_total = []
        train_losses_task = []
        train_losses_domain = []
        train_losses_r01b = []    
        train_losses_r01b_ranking = []
            
        for epoch in range(self.num_epochs):
            
            self.train()
            # Mini-batch training
            seed = 42 + epoch
            shuffled_indices = torch.randperm(self.X_train_tensor.size(0))
            self.X_train_tensor = self.X_train_tensor[shuffled_indices]
            self.y_train_tensor = self.y_train_tensor[shuffled_indices]
            self.d_train_tensor = self.d_train_tensor[shuffled_indices]
            self.sampleid_train = self.sampleid_train[shuffled_indices]
            
            shuffled_indices_r01b = torch.randperm(X_r01b_cancer_tensor.size(0))
            X_r01b_cancer_tensor_shuffled = X_r01b_cancer_tensor[shuffled_indices_r01b]
            y_r01b_cancer_tensor_shuffled = y_r01b_cancer_tensor[shuffled_indices_r01b]
            
            for batch_start in range(0, len(self.X_train_tensor), self.batch_size):
                batch_end = batch_start + self.batch_size
                ith = batch_start // self.batch_size
                p = (ith + epoch * num_iterations) / (self.num_epochs * num_iterations)
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                
                batch_X = self.X_train_tensor[batch_start:batch_end]
                batch_y = self.y_train_tensor[batch_start:batch_end]
                batch_d = self.d_train_tensor[batch_start:batch_end]

                indice_batch_healthy = torch.where(batch_y == 0)
                batch_X_healthy = batch_X[indice_batch_healthy]
                batch_y_healthy = batch_y[indice_batch_healthy]
                
                healthy_num = min(70, batch_X_healthy.size(0))
                batch_X_healthy = batch_X_healthy[0:healthy_num]
                batch_y_healthy = batch_y_healthy[0:healthy_num]
                
              
                X_r01b_bind_tensor = torch.cat((X_r01b_cancer_tensor_shuffled, batch_X_healthy), dim=0)
                y_r01b_bind_tensor = torch.cat((y_r01b_cancer_tensor_shuffled, batch_y_healthy), dim=0)
                
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_d = batch_d.to(device)
                
                X_r01b_bind_tensor = X_r01b_bind_tensor.to(device)
                y_r01b_bind_tensor = y_r01b_bind_tensor.to(device)
                
                # batch_X_source = batch_X[batch_d == 0]   # 0 being the large cluster
                # batch_y_source = batch_y[batch_d == 0]
                
                # batch_X_source = batch_X_source.to(device)
                # batch_y_source = batch_y_source.to(device)
                
                ### Forward pass
                # outputs_task, _ = self(batch_X_source, alpha)
                # outputs_task, _ = self(batch_X, alpha)
                outputs_task, outputs_domain, outputs_r01b = self(batch_X, X_r01b_bind_tensor, alpha)
                
                outputs_r01b_cancer = outputs_r01b[0:healthy_num]
                outputs_r01b_healthy = outputs_r01b[70:(70+healthy_num)]
                ones_tensor = torch.ones(healthy_num).to(device)
                
                # calculate task and domain loss
                # loss_task = self.criterion_task(outputs_task, batch_y_source)
                loss_task = self.criterion_task(outputs_task, batch_y)
                loss_domain = self.criterion_domain(outputs_domain, batch_d)
                loss_r01b = self.criterion_r01b(outputs_r01b, y_r01b_bind_tensor)
                loss_r01b_ranking = self.criterion_r01b_ranking(outputs_r01b_cancer, outputs_r01b_healthy, ones_tensor)
                
                loss = loss_task + self.loss_lambda * loss_domain + self.gamma_r01b * loss_r01b + self.gamma_r01b * loss_r01b_ranking * 2
                
                ### to plot the loss by iteration
                train_losses_total.append((loss.item())/500.0)
                train_losses_task.append(loss_task.item())
                train_losses_domain.append(loss_domain.item())
                train_losses_r01b.append(loss_r01b.item())
                train_losses_r01b_ranking.append(loss_r01b_ranking.item())
                
                # Real-time loss plotting
                # plt.plot(train_losses_total, label='Total Loss')
                # plt.plot(train_losses_task, label='Task Loss')
                # plt.plot(train_losses_domain, label='Domain Loss')
                # plt.plot(train_losses_r01b, label='R01b Loss')
                # plt.xlabel('Iteration')
                # plt.ylabel('Loss')
                # plt.legend()
                # plt.title('Training Loss Components in Real-Time')
                # plt.pause(0.1)  # Add a short pause to allow the plot to update
                # plt.clf()
                
                
                ##### source domain
                # Zero parameter gradients
                self.optimizer_extractor.zero_grad()
                self.optimizer_task.zero_grad()
                self.optimizer_domain.zero_grad()
                self.optimizer_r01b.zero_grad()
                             
                # Backward and optimize
                loss.backward()
                self.optimizer_extractor.step()
                self.optimizer_task.step()
                self.optimizer_domain.step()
                self.optimizer_r01b.step()
                 
                 
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
                
                test_outputs,_,_=self(self.X_test_tensor, X_r01b_bind_tensor, alpha=0.1)
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
                    if epochs_without_improvement >= self.patience:
                        print(f"Early stopping triggered! No improvement in {self.patience} epochs.")
                        break
        
        self.train()
        self.max_test_auc = max_test_auc
        self.load_state_dict(best_model)
        
        ### plot the final loss curves
        # plt.plot(train_losses_total, label='Total Loss')
        # plt.plot(train_losses_task, label='Task Loss')
        # plt.plot(train_losses_domain, label='Domain Loss')
        # plt.plot(train_losses_r01b, label='R01b Loss')
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.title('Final Training Loss Components')
        # plt.savefig(f"{output_path}/train_and_tune_1D_loss_cluster{self.selected_cluster}.png")
        # plt.show()
        
        
        # export the best model
        if not os.path.exists(f"{output_path}/Raw/"):
            os.makedirs(f"{output_path}/Raw/")
        torch.save(self,f"{output_path}/Raw/{self.feature_type}_DANN_best__{self.methods}_cluster{self.selected_cluster}.pt")
        
        # obtain scores of all samples and export
        with torch.no_grad():
            self.eval()

            self.X_all_tensor = self.X_all_tensor.to(device)
            outputs_all,_,_ = self(self.X_all_tensor,None,alpha=0.1)
            outputs_all = outputs_all.to("cpu")

            self.data_idonly['DANN_score'] = outputs_all.detach().cpu().numpy()
            self.data_idonly.to_csv(f"{output_path}/Raw/{self.feature_type}_score_{self.methods}_cluster{self.selected_cluster}.csv", index=False)
            
            ### calculate R01B sensitivity at training 98% specificity    
            training_score = self.data_idonly.loc[self.data_idonly["train"] == "training","DANN_score"]
            training_response = self.data_idonly.loc[self.data_idonly["train"] == "training","Train_Group"] 
            threshold98 = find_threshold(training_response, training_score, 0.98)
            self.training_auc = roc_auc_score(training_response, training_score)
            
            testing_score = self.data_idonly.loc[self.data_idonly["train"] == "testing","DANN_score"]
            testing_repsonse = self.data_idonly.loc[self.data_idonly["train"] == "testing","Train_Group"] 
            self.testing_sens = find_sensitivity(testing_repsonse, testing_score, threshold98)
            
            print("==================== DANN score =====================")
            print(f"Training AUC: {self.training_auc:.4f}, threshold 98: {threshold98:.4f}, R01B sensitivity: {self.testing_sens:.4f}")
            print("=====================================================")
                
        # fine tuning with R01BMatch data
        if(self.R01BTuning and R01BTuning_fit):
            self.train()
            optimizer_R01B = torch.optim.Adam(self.parameters(), lr=1e-6)

            # Perform forward pass and compute loss
            self.X_train_tensor_R01B = self.X_train_tensor_R01B.to(device)
            self.y_train_tensor_R01B = self.y_train_tensor_R01B.to(device)

            for epoch_toupdate in range(30):
                outputs_R01B,_,_ = self(self.X_train_tensor_R01B, None, alpha = 0.1)
                loss = self.criterion_task(outputs_R01B, self.y_train_tensor_R01B)

                # Backpropagation and parameter update
                optimizer_R01B.zero_grad()
                loss.backward()
                optimizer_R01B.step()
            
            if not os.path.exists(f"{output_path}/R01BTuned/"):
                os.makedirs(f"{output_path}/R01BTuned/")    
            torch.save(self,f"{output_path}/R01BTuned/{self.feature_type}_DANN_best_R01BTuned_{self.methods}_cluster{self.selected_cluster}.pt")
            
            with torch.no_grad():
                self.eval()
                
                self.X_test_tensor=self.X_test_tensor.to(device)
                self.y_test_tensor=self.y_test_tensor.to(device)
                
                test_outputs,_,_=self(self.X_test_tensor,None,alpha=0.1)
                test_outputs=test_outputs.to("cpu")

                test_loss=self.criterion_task(test_outputs, self.y_test_tensor.to("cpu"))
                test_auc = roc_auc_score(self.y_test_tensor.to("cpu"), test_outputs.to("cpu"))
                print(f"Test AUC (tuned): {test_auc.item():.4f}, Test Loss (tuned): {test_loss.item():.4f}")
                print("*********************")
                
                ### obtain scores of all samples
                self.X_all_tensor = self.X_all_tensor.to(device)
                outputs_all_tuned,_,_ = self(self.X_all_tensor,None,alpha=0.1)
                outputs_all_tuned = outputs_all_tuned.to("cpu")

            self.data_idonly['DANN_score_tuned'] = outputs_all_tuned.detach().cpu().numpy()
            self.data_idonly.to_csv(f"{output_path}/R01BTuned/{self.feature_type}_score_R01BTuned_{self.methods}_cluster{self.selected_cluster}.csv", index=False)
            
            ### calculate R01B sensitivity at training 98% specificity    
            training_score_tuned = self.data_idonly.loc[self.data_idonly["train"] == "training","DANN_score_tuned"]
            training_response = self.data_idonly.loc[self.data_idonly["train"] == "training","Train_Group"] 
            threshold98 = find_threshold(training_response, training_score_tuned, 0.98)
            self.training_auc_tuned = roc_auc_score(training_response, training_score_tuned)
            
            testing_score_tuned = self.data_idonly.loc[self.data_idonly["train"] == "testing","DANN_score_tuned"]
            testing_repsonse = self.data_idonly.loc[self.data_idonly["train"] == "testing","Train_Group"] 
            self.testing_sens_tuned = find_sensitivity(testing_repsonse, testing_score_tuned, threshold98)
            
            print("============ DANN score (fine tuned) ===============")
            print(f"Training AUC: {self.training_auc_tuned:.4f}, threshold 98: {threshold98:.4f}, R01B sensitivity: {self.testing_sens_tuned:.4f}")
            print("====================================================")
         
    
    def predict(self, X_predict_tensor, y_predict_tensor):
        
        X_predict_tensor = X_predict_tensor.to(self.device)
        y_predict_tensor = y_predict_tensor.to(self.device)
        with torch.no_grad():
            self.eval()
            outputs_predict,_,_ = self(X_predict_tensor,None,alpha=0.1)        
        return(outputs_predict.detach().cpu().numpy())
            
    def crossvalidation(self,num_folds, output_path):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=99)
        
        fold_scores = []  # List to store validation scores
        fold_labels = []
        fold_numbers = []
        fold_sampleid = []
        
        # self.X_r01b_tensor=self.X_r01b_tensor.to(device)
        # self.y_r01b_tensor=self.y_r01b_tensor.to(device)
        self.patience = 50
        indice_r01b_cancer = torch.where(self.y_r01b_tensor == 1)
        X_r01b_cancer_tensor = self.X_r01b_tensor[indice_r01b_cancer]
        y_r01b_cancer_tensor = self.y_r01b_tensor[indice_r01b_cancer]
        
        for fold, (train_index, val_index) in enumerate(kf.split(self.X_train_tensor)):
            X_train_fold, X_val_fold = self.X_train_tensor[train_index], self.X_train_tensor[val_index]
            y_train_fold, y_val_fold = self.y_train_tensor[train_index], self.y_train_tensor[val_index]
            d_train_fold = self.d_train_tensor[train_index]
            sampleid_val_fold = self.sampleid_train[val_index]
            
            ### reset the model
            self.weight_reset()
            self.to(device)
            
            num_iterations = (X_train_fold.size(0) // self.batch_size) + 1          # get the iteration number
            
            # patience = 50
            max_test_auc = 0.0
            best_model_cv = None
            epochs_without_improvement = 0
         
            for epoch in range(self.num_epochs):
                shuffled_indices = torch.randperm(X_train_fold.shape[0])
                X_train_fold = X_train_fold[shuffled_indices]
                y_train_fold = y_train_fold[shuffled_indices]
                d_train_fold = d_train_fold[shuffled_indices]
                
                shuffled_indices_r01b = torch.randperm(X_r01b_cancer_tensor.size(0))
                X_r01b_cancer_tensor_shuffled = X_r01b_cancer_tensor[shuffled_indices_r01b]
                y_r01b_cancer_tensor_shuffled = y_r01b_cancer_tensor[shuffled_indices_r01b]
                
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
                                                                                      
                    indice_batch_healthy = torch.where(batch_y == 0)
                    batch_X_healthy = batch_X[indice_batch_healthy]
                    batch_y_healthy = batch_y[indice_batch_healthy]
                    
                    healthy_num = min(70, batch_X_healthy.size(0))
                    batch_X_healthy = batch_X_healthy[0:healthy_num]
                    batch_y_healthy = batch_y_healthy[0:healthy_num]
                    X_r01b_bind_tensor = torch.cat((X_r01b_cancer_tensor_shuffled, batch_X_healthy), dim=0)
                    y_r01b_bind_tensor = torch.cat((y_r01b_cancer_tensor_shuffled, batch_y_healthy), dim=0)
                    
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    batch_d = batch_d.to(device)
                    
                    X_r01b_bind_tensor = X_r01b_bind_tensor.to(device)
                    y_r01b_bind_tensor = y_r01b_bind_tensor.to(device)
                    
                    # batch_X_source = batch_X[batch_d == 0]
                    # batch_y_source = batch_y[batch_d == 0]

                    # if batch_X_source.shape[0] <= 0:
                    #     print("=======   not enough doamin 0 samples in batch_X   ========")
                    #     continue

                    # batch_X_source = batch_X_source.to(device)
                    # batch_y_source = batch_y_source.to(device)
                                       
                    # outputs_task, _ = self(batch_X_source,alpha)
                    outputs_task, outputs_domain, outputs_r01b = self(batch_X, X_r01b_bind_tensor, alpha)
                    
                    outputs_r01b_cancer = outputs_r01b[0:healthy_num]
                    outputs_r01b_healthy = outputs_r01b[70:(70+healthy_num)]
                    ones_tensor = torch.ones(healthy_num).to(device)
                    
                    # subset domain outputs to only lung cancer samples
                    outputs_domain_lung = outputs_domain[batch_y == 1]
                    batch_d_lung = batch_d[batch_y == 1]
                    
                    # loss_task = self.criterion_task(outputs_task, batch_y_source)
                    loss_task = self.criterion_task(outputs_task, batch_y)
                    loss_domain = self.criterion_domain(outputs_domain_lung, batch_d_lung)
                    loss_r01b = self.criterion_r01b(outputs_r01b, y_r01b_bind_tensor)
                    loss_r01b_ranking = self.criterion_r01b_ranking(outputs_r01b_cancer, outputs_r01b_healthy, ones_tensor)
                    
                    loss = loss_task + self.loss_lambda * loss_domain + self.gamma_r01b * loss_r01b + self.gamma_r01b * loss_r01b_ranking * 2
                    
                    self.optimizer_extractor.zero_grad()
                    self.optimizer_task.zero_grad()
                    self.optimizer_domain.zero_grad()
                    self.optimizer_r01b.zero_grad()
                    
                    loss.backward()
                    self.optimizer_extractor.step()
                    self.optimizer_task.step()
                    self.optimizer_domain.step()
                    self.optimizer_r01b.step()
                    
                # train_auc = roc_auc_score(
                #     batch_y_source.to('cpu').detach().numpy(), outputs_task.to('cpu').detach().numpy()
                # )
                print(f"Fold: {fold+1}/{num_folds}, Epoch: {epoch+1}/{self.num_epochs}, i: {batch_start//self.batch_size}")
                print(f"Train total loss: {loss.item():.4f}, Train task loss: {loss_task.item():.4f}")
                print("-------------------------")
        
                with torch.no_grad():
                    self.eval()
                    val_outputs, _, _ = self(X_val_fold.to(device), None, alpha=0.1)
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
                        if epochs_without_improvement >= self.patience:
                            print(f"Early stopping triggered for Fold {fold+1}! No improvement in {self.patience} epochs.")
                            break
            
            self.load_state_dict(best_model_cv)
            
            if not os.path.exists(f"{output_path}/cv/"):
                os.makedirs(f"{output_path}/cv/")
             
            torch.save(self, f"{output_path}/cv/{self.feature_type}_DANN_cv_fold{fold+1}_{self.methods}_cluster{self.selected_cluster}.pt")
            fold_scores.append(val_outputs.detach().cpu().numpy())  # Collect validation scores for the fold
            fold_labels.append(y_val_fold.detach().cpu().numpy())
            fold_numbers.append(np.repeat(fold+1, len(y_val_fold.detach().cpu().numpy())))
            fold_sampleid.append(sampleid_val_fold)
                        
        all_scores = np.concatenate(fold_scores)
        all_labels = np.concatenate(fold_labels)
        all_numbers = np.concatenate(fold_numbers)
        all_sampleid = np.concatenate(fold_sampleid)
        
        # Save fold scores to CSV file
        df = pd.DataFrame({'Fold': all_numbers,
                        'Scores': all_scores,
                        'Train_Group': all_labels,
                        'SampleID': all_sampleid})
        
        df.to_csv(f"{output_path}/cv/{self.feature_type}_CV_score_{self.methods}_cluster{self.selected_cluster}.csv", index=False)
                        
                                   
                    
        
        
