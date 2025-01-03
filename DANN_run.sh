#!/bin/bash
feature_type_list=(AE Frag Arm Cnv Griffin MCMS Ma)
input_size=(760 1250 1000 2550 2650 350 50)
output_path=./Output_example/
data_dir=./Feature_example/feature_example.pkl
R01BTune=No
cluster_method_list=(None kmeans DBSCAN GMM MeanShift)
nfold=5
for i in {0..0}
do
    for j in {0..0}
    do
        python ./DANN_main.py ${feature_type_list[i]} 1D ${input_size[i]} 100 1000 ${output_path} ${data_dir} ${R01BTune} ${cluster_method_list[j]} ${nfold}
    done
done

