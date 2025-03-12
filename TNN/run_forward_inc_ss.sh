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

