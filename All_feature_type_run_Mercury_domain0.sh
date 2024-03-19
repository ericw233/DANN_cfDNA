#!/bin/bash
feature_type_list=(AE AE_Frag AE_Arm AE_Cnv AE_Griffin Frag Arm Griffin Cnv MCMS Gemini Ma)
input_size=(550 64 64 64 64 1200 1000 2650 2550 350 120 50)
output_path=/mnt/binf/eric/DANN_Jan2024Results_new/DANN_0227_AE
data_dir=/mnt/binf/eric/Mercury_Dec2023/Feature_all_Dec2023_AE_v5.pkl
R01BTune=No
cluster_method_list=(None kmeans DBSCAN GMM MeanShift)
nfold=5
for i in {5..11}
do
    for j in {0..4}
    do
        python ./DANN_run.py ${feature_type_list[i]} 1D ${input_size[i]} 100 1000 ${output_path} ${data_dir} ${R01BTune} ${cluster_method_list[j]} ${nfold}
    done
done

