!/usr/bin/env bash

source /home/vpatro/anaconda3/etc/profile.d/conda.sh
conda activate dnn_env

###############
### TRAIN FORWARD DNNS
###############

### INCONEL DATA
python train_forward.py inconel standard configs/forward/forward.yaml min_max --mode train  

### STAINLESS STEEL DATA
python train_forward.py stainless_steel standard configs/forward/forward.yaml min_max --mode train

### INCONEL TASK WITH STAINLESS STEEL PARTIAL HOT START
python train_forward.py inconel transfer_learning configs/forward/forward.yaml min_max --hot_start_dataset stainless_steel --num_layers_to_transfer 2 --hot_start_type partial

### STAINLESS STEEL TASK WITH INCONEL PARTIAL HOT START
python train_forward.py stainless_steel transfer_learning configs/forward/forward.yaml min_max --hot_start_dataset inconel --num_layers_to_transfer 2 --hot_start_type partial

# ### INCONEL TASK WITH STAINLESS STEEL FULL HOT START
python train_forward.py inconel transfer_learning configs/forward/forward.yaml min_max --hot_start_dataset stainless_steel --num_layers_to_transfer 2 --hot_start_type full

# ### STAINLESS STEEL TASK WITH INCONEL FULL HOT START
python train_forward.py stainless_steel transfer_learning configs/forward/forward.yaml min_max --hot_start_dataset inconel --num_layers_to_transfer 2 --hot_start_type full

# ### INCONEL TASK WITH STAINLESS STEEL PARTIAL HOT START
python train_forward.py inconel transfer_learning configs/forward/forward.yaml min_max --hot_start_dataset stainless_steel --num_layers_to_transfer 1 --hot_start_type partial

# ### STAINLESS STEEL TASK WITH INCONEL PARTIAL HOT START
python train_forward.py stainless_steel transfer_learning configs/forward/forward.yaml min_max --hot_start_dataset inconel --num_layers_to_transfer 1 --hot_start_type partial

# ### INCONEL TASK WITH STAINLESS STEEL FULL HOT START
python train_forward.py inconel transfer_learning configs/forward/forward.yaml min_max --hot_start_dataset stainless_steel --num_layers_to_transfer 1 --hot_start_type full

# ### STAINLESS STEEL TASK WITH INCONEL FULL HOT START
python train_forward.py stainless_steel transfer_learning configs/forward/forward.yaml min_max --hot_start_dataset inconel --num_layers_to_transfer 1 --hot_start_type full




