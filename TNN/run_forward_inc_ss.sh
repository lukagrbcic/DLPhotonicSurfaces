!/usr/bin/env bash

source /home/vpatro/anaconda3/etc/profile.d/conda.sh
conda activate dnn_env

###############
### TRAIN FORWARD DNNS
###############

### INCONEL DATA
python train_forward.py inconel standard configs/forward/forward.yaml --mode train  

### STAINLESS STEEL DATA
python train_forward.py stainless_steel standard configs/forward/forward.yaml --mode train

### INCONEL TASK WITH STAINLESS STEEL HOT START
python train_forward.py inconel transfer_learning configs/forward/forward.yaml --hot_start_dataset stainless_steel --num_layers_to_transfer 1

### STAINLESS STEEL TASK WITH INCONEL HOT START
python train_forward.py stainless_steel transfer_learning configs/forward/forward.yaml --hot_start_dataset inconel --num_layers_to_transfer 1

