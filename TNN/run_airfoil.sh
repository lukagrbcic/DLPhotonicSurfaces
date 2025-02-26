!/bin/bash

source /home/vpatro/anaconda3/etc/profile.d/conda.sh
conda activate dnn_env
###############
### TRAIN FORWARD DNNS
###############

### airfoil_re_1_3 DATA
python train_forward.py airfoil_re_1_3 configs/forward/forward.yaml 

### airfoil_re_3_6 DATA
python train_forward.py airfoil_re_3_6 configs/forward/forward.yaml 


###############
### TRAIN INVERSE DNNS
###############

# ### airfoil_re_1_3 inverse will be trained and saved
python main.py standard airfoil_re_1_3 airfoil_re_1_3 

# ### airfoil_re_3_6 inverse will be trained and saved
python main.py standard airfoil_re_3_6 airfoil_re_3_6

###############
### TL Experiments
###############

#### transfer learning experiments on airfoil_re_1_3
python main.py transfer_learning airfoil_re_1_3 airfoil_re_1_3 --inverse_DNN_dataset airfoil_re_3_6
python main.py transfer_learning airfoil_re_1_3 airfoil_re_3_6 
python main.py transfer_learning airfoil_re_1_3 airfoil_re_3_6 --inverse_DNN_dataset airfoil_re_3_6


#### transfer learning experiments on airfoil_re_3_6
python main.py transfer_learning airfoil_re_3_6 airfoil_re_3_6 --inverse_DNN_dataset airfoil_re_1_3
python main.py transfer_learning airfoil_re_3_6 airfoil_re_1_3 
python main.py transfer_learning airfoil_re_3_6 airfoil_re_1_3 --inverse_DNN_dataset airfoil_re_1_3