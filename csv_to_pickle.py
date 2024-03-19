import pandas as pd
import numpy as np

# data_tmp = pd.read_csv("/mnt/binf/eric/Mercury_Dec2023/Feature_all_Dec2023_Lungv2.csv")
# data_tmp.to_pickle("/mnt/binf/eric/Mercury_Dec2023/Feature_all_Dec2023_Lungv2.pkl")

pkl_tmp = pd.read_pickle("/mnt/binf/eric/Mercury_Dec2023/Feature_all_Dec2023_withAE.pkl")

# tmp = pkl_tmp.loc[pkl_tmp['R01B_label'] == 'R01B_match',["SampleID","R01B_label"]]
# tmp.to_csv("/mnt/binf/eric/Mercury_Dec2023/R01B_label.csv")

colname = pkl_tmp.columns    
print(len([col for col in colname if "Ma" in col]))

sampleinfo = pd.read_csv("/mnt/binf/eric/Mercury_Dec2023/Info/Test1.all.full.info.list",sep='\t',low_memory=False)
pkl_tmp_anno = pd.merge(pkl_tmp, sampleinfo.loc[:,["SampleID","GroupLevel2"]], on = ["SampleID"], how = "inner")


def prepare_singlecancer_feature(cancertype = "Lung"):

    pkl_tmp_tmp = pkl_tmp_anno.copy()
    pkl_tmp_tmp['Select'] = "No"
    pkl_tmp_tmp.loc[(pkl_tmp_tmp["train"] == "training") & (pkl_tmp_tmp["GroupLevel2"] == cancertype),"Select"] = "Yes"
    rows_to_change = pkl_tmp_tmp.loc[(pkl_tmp_tmp["train"] == "training") & (pkl_tmp_tmp["Project"] == "KAG9")].index[:1000]
    pkl_tmp_tmp.loc[rows_to_change,"Select"] = "Yes"
    
    pkl_tmp_tmp.loc[(pkl_tmp_tmp['train'] == "training") & (pkl_tmp_tmp["Select"] == "No"),"train"] = "training-heldout"
    pkl_tmp_tmp["Domain"] = pkl_tmp_tmp["Domain"].astype(int)

    print(pd.crosstab(pkl_tmp_tmp["Train_Group"],pkl_tmp_tmp["train"]))
    pkl_tmp_tmp.to_pickle(f"/mnt/binf/eric/Mercury_Dec2023/Feature_all_Dec2023_withAE_{cancertype}.pkl")


cancertype_list = ["Lung", "Colorectal", "Gastro", "Breast", "Prostate", "Liver"]

for cancertype_tmp in cancertype_list:
    prepare_singlecancer_feature(cancertype = cancertype_tmp)

