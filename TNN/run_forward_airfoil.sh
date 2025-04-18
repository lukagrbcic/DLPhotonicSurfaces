!/usr/bin/env bash

source /home/vpatro/anaconda3/etc/profile.d/conda.sh
conda activate dnn_env

###############
### TRAIN FORWARD DNNS
###############

### AIRFOIL RE_1_3 DATA
python train_forward.py airfoil_re_1_3 standard configs/forward/forward.yaml --mode train 