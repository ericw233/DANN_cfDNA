#!/bin/bash
feature_type_list=(Frag Arm Griffin Cnv MCMS)
output_path=/mnt/binf/eric/DANN_AugResults_MGI/DANN_modelL2_0803v2
data_dir=/mnt/binf/eric/Mercury_Aug2023_new/Feature_MGI_SeqDomain.csv
input_size=(1200 950 2600 2500 200)
for i in {0..4}
do
    python ./DANN_run.py ${feature_type_list[i]} 1D ${input_size[i]} 200 200 ${output_path} ${data_dir} False
done

