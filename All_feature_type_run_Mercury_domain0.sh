#!/bin/bash
feature_type_list=(Frag Arm Griffin Cnv MCMS Ma)
input_size=(1200 950 2600 2500 200 21)
output_path=/mnt/binf/eric/DANN_AugResults/DANN_1D_DomainAll0_impute_0815
data_dir=/mnt/binf/eric/Mercury_Aug2023_new/Feature_all_Aug2023_Domain0_SeqBatchAll.csv
R01BTune=Yes
for i in {0..5}
do
    python ./DANN_run.py ${feature_type_list[i]} 1D ${input_size[i]} 100 500 ${output_path} ${data_dir} ${R01BTune}
done

