!/usr/bin/env bash

source /home/vpatro/anaconda3/etc/profile.d/conda.sh
conda activate dnn_env

###############
### TRAIN INVERSE DNNS
###############

########################################################
# STANDARD CONFIG: NO TRANSFER LEARNING FOR INVERSE DNN
########################################################

## for inconel task: training from scratch an inverse DNN with a
## forward DNN trained from SCRATCH on inconel

# python main.py config dataset fwd_dset fwd_hot_start
python main.py standard inconel inconel --forward_DNN_from_scratch --mode train

### For inconel task: training from scratch an inverse DNN with a 
### forward DNN (trained on inconel task) that was hot started on stainless steel
python main.py standard inconel inconel --forward_DNN_hot_start --forward_DNN_hot_start_dataset stainless_steel --mode train --num_forward_layers_transferred 1

# # # ### for stainless steel task: training from scratch an inverse DNN with a
# # # ### forward DNN trained from SCRATCH on stainless steel
python main.py standard stainless_steel stainless_steel --forward_DNN_from_scratch --mode train

# # # ### For inconel task: training from scratch an inverse DNN with a 
# # # ### forward DNN (trained on inconel task) that was hot started on stainless steel
python main.py standard stainless_steel stainless_steel --forward_DNN_hot_start --forward_DNN_hot_start_dataset inconel --mode train --num_forward_layers_transferred 1


# ###############
# ### TL Experiments
# ###############

# ### for inconel task
# ### forward DNN starting from scratch
# ### inverse DNN hot started on SS
python main.py transfer_learning inconel inconel --forward_DNN_from_scratch --inverse_DNN_hot_start_dataset stainless_steel \
    --mode train --num_inverse_layers_to_transfer 1

# # ### for inconel task
# # ### forward DNN hot started on SS
# # ### inverse DNN hot started on SS
python main.py transfer_learning inconel inconel --forward_DNN_hot_start --forward_DNN_hot_start_dataset stainless_steel \
    --inverse_DNN_hot_start_dataset stainless_steel --mode train --num_inverse_layers_to_transfer 1 --num_forward_layers_transferred 1

# # ### for ss task
# # ### forward DNN starting from scratch
# # ### inverse DNN hot started on inconel
python main.py transfer_learning stainless_steel stainless_steel --forward_DNN_from_scratch --inverse_DNN_hot_start_dataset inconel \
    --mode train --num_inverse_layers_to_transfer 1

# # ### for ss task
# # ### forward DNN hot started on inconel
# # ### inverse DNN hot started on inconel
python main.py transfer_learning stainless_steel stainless_steel --forward_DNN_hot_start --forward_DNN_hot_start_dataset inconel \
    --inverse_DNN_hot_start_dataset inconel --mode train --num_inverse_layers_to_transfer 1 --num_forward_layers_transferred 1





# # Transfer learning experiments on Inconel
# # python main.py transfer_learning inconel stainless_steel
# # for num_layers in 2 4 6; do
# #     python main.py transfer_learning inconel inconel --inverse_DNN_dataset stainless_steel --num_inverse_layers_frozen $num_layers
# #     python main.py transfer_learning inconel stainless_steel --inverse_DNN_dataset stainless_steel --num_inverse_layers_frozen $num_layers
# # done

# # # Transfer learning experiments on Stainless Steel
# # python main.py transfer_learning stainless_steel inconel
# # for num_layers in 2 4 6; do
# #     python main.py transfer_learning stainless_steel stainless_steel --inverse_DNN_dataset inconel --num_inverse_layers_frozen $num_layers
# #     python main.py transfer_learning stainless_steel inconel --inverse_DNN_dataset inconel --num_inverse_layers_frozen $num_layers
# # done