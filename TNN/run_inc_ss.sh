!/usr/bin/env bash

source /home/vpatro/anaconda3/etc/profile.d/conda.sh
conda activate dnn_env
###############
### TRAIN FORWARD DNNS
###############

### INCONEL DATA
# python train_forward.py inconel configs/forward/forward.yaml 

### STAINLESS STEEL DATA
# python train_forward.py stainless_steel configs/forward/forward.yaml 


###############
### TRAIN INVERSE DNNS
###############

# ### inconel inverse will be trained and saved
python main.py standard inconel inconel 

# ### stainless steel inverse will be trained and saved
python main.py standard stainless_steel stainless_steel

###############
### TL Experiments
###############

# Transfer learning experiments on Inconel
# python main.py transfer_learning inconel stainless_steel
# for num_layers in 2 4 6; do
#     python main.py transfer_learning inconel inconel --inverse_DNN_dataset stainless_steel --num_inverse_layers_frozen $num_layers
#     python main.py transfer_learning inconel stainless_steel --inverse_DNN_dataset stainless_steel --num_inverse_layers_frozen $num_layers
# done

# Transfer learning experiments on Stainless Steel
# python main.py transfer_learning stainless_steel inconel
# for num_layers in 2 4 6; do
#     python main.py transfer_learning stainless_steel stainless_steel --inverse_DNN_dataset inconel --num_inverse_layers_frozen $num_layers
#     python main.py transfer_learning stainless_steel inconel --inverse_DNN_dataset inconel --num_inverse_layers_frozen $num_layers
# done