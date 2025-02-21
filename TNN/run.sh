
!/bin/bash

source /home/vpatro/anaconda3/etc/profile.d/conda.sh
conda activate dnn_env
###############
### TRAIN FORWARD DNNS
###############

### INCONEL DATA
python train_forward.py inconel configs/forward/forward.yaml 

### STAINLESS STEEL DATA
python train_forward.py stainless_steel configs/forward/forward.yaml 


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

# ### transfer learning experiments
python main.py transfer_learning inconel inconel --inverse_DNN_dataset stainless_steel
python main.py transfer_learning inconel stainless_steel 
python main.py transfer_learning inconel stainless_steel --inverse_DNN_dataset stainless_steel

python main.py transfer_learning stainless_steel stainless_steel --inverse_DNN_dataset inconel
python main.py transfer_learning stainless_steel inconel 
python main.py transfer_learning stainless_steel inconel --inverse_DNN_dataset inconel