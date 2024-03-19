from AE_transform_feature import AE_transform
import pandas as pd
import numpy as np

data_dir = "/mnt/binf/eric/Mercury_Dec2023/Feature_all_Dec2023_v5.csv"
feature_list = ["Frag", "Arm", "Cnv", "Griffin", "MCMS", "Ma","Gemini"]
input_size_list = [1100,950,2500,2600,180,25,60]

if(data_dir.endswith('.csv')):
    data = pd.read_csv(data_dir)
elif(data_dir.endswith('.pkl')):
    data = pd.read_pickle(data_dir)
    
data_idonly = data.loc[:,['SampleID','Train_Group','train','Project','R01B_label']]
data_AE = data_idonly

for i in range(len(feature_list)):
      
    if input_size_list[i] >= 500:
        data_tmp = AE_transform(data_dir=data_dir,input_size=input_size_list[i],feature_type=feature_list[i], encoding_size=64, control="No")
        data_tmp = data_tmp.drop(columns=['SampleID','Train_Group','train','Project','R01B_label'])
        data_AE = pd.concat([data_AE, data_tmp], axis=1)
        
    else:
        data_tmp = data.filter(regex = feature_list[i], axis=1).add_prefix("AE_raw_")
        data_AE = pd.concat([data_AE, data_tmp], axis=1)
        

data_full = pd.merge(data,data_AE,on=['SampleID','Train_Group','train','Project','R01B_label'],how="inner")
        
print(data_AE.shape)
print(data.shape)
   
data_full.to_pickle("/mnt/binf/eric/Mercury_Dec2023/Feature_all_Dec2023_AE_v5.pkl") 

### drop AE_raw columns

# AE_raw_columns = [colname for colname in data_full.columns if 'AE_raw_' in colname]
# data_full_norep = data_full.drop(columns=AE_raw_columns)

# data_full_norep.to_pickle("/mnt/binf/eric/Mercury_Dec2023/Feature_all_Dec2023_withAE_norep.pkl") 



    


