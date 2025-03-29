import sys
import os
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import time
sys.path.insert(0, 'src')
sys.path.insert(0, '../..')
import DLPhotonicSurfaces.TNN.dnn as invfow
import tnn as tnn
from load_data import get_data_paths_for_tnn_training, load_data_for_tnn_training

import argparse
import pandas as pd
import json
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('configuration',type=str,
        help='enter whether you are performing the standard TNN configuration (learning inverse from scratch) \
             or Transfer Learning (starting the inverse DNN off with pretrained weights)')

    parser.add_argument('dataset_name', type=str, 
        help='enter the name of the dataset we are doing predictive tasks on')

    parser.add_argument('forward_DNN_dataset', type=str,
        help='enter the name of the dataset (the task) the forward DNN was trained on')

    parser.add_argument('--forward_DNN_hot_start', dest='forward_DNN_hot_start', action='store_true',
        help = 'forward DNN has been hot started')

    parser.add_argument('--forward_DNN_from_scratch', dest='forward_DNN_hot_start', action='store_false',
        help='forward DNN has not been hot started and was trained from scratch')

    parser.add_argument('--forward_DNN_hot_start_dataset', type=str, default='from_scratch',
        help='enter the name of the dataset the forward DNN was hot started with')

    parser.add_argument('--inverse_DNN_hot_start_dataset', type=str, default='from_scratch',
        help='enter the name of the dataset we will hot start the inverse DNN with')

    parser.add_argument('--mode', type=str, default = 'train',
        help = 'enter train to do training followed by inference and enter inference to do inference on a pretrained model')

    parser.add_argument('--num_inverse_layers_to_transfer', type=int, default=0,
        help='enter how many layer of the inverse DNN should be transferred and frozen in the TL configuration')

    parser.add_argument('--num_forward_layers_transferred', type=int, default=0,
        help='enter how many layers of the forward DNN were used in hot starting')

    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data_for_tnn_training(dataset_name=args.dataset_name)
    print('\n -------------------- \n')
    print(f'LOADED {args.dataset_name} DATASET')
    print('\n -------------------- \n')

    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    print(f'Input size: {input_size} (emissivity), Output size: {output_size} (laser parameters) \n')

    print(f'shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}')
    print(f'shape of X_test: {X_test.shape}, shape of y_test: {y_test.shape} \n')


    forward_architecture = invfow.forwardMLP(output_size, input_size).to(device)
    inverse_architecture = invfow.inverseMLP(input_size, output_size).to(device)
    
    from model_check import forward_DNN_check

    forward_DNN = forward_DNN_check(args)

    # no transfer learning configuration, inverse DNN weights initialized from scratch
    if args.configuration == 'standard':

        from model_check import inverse_DNN_standard_config_check
        inverse_DNN_standard_config_check(args)        

        time.sleep(2)

    elif args.configuration == 'transfer_learning': # transfer learning configuration
        ### if we don't give an inverse_DNN_hot_start_dataset (ie don't want to load a pretrained inverse DNN), inverse_DNN will be set to None in the tnn
            
        from model_check import inverse_DNN_tl_config_check
        inverse_DNN_tl_config_check(args)

        time.sleep(2)

    ##### GENERAL CHECKS

    #### forward has to map the correct task
    assert args.dataset_name == args.forward_DNN_dataset

    if args.forward_DNN_hot_start:
        ## make sure we have hot started with the dataset that isn't the one we're currently training on
        assert args.forward_DNN_dataset != args.forward_DNN_hot_start_dataset

    max_epochs = 1000
    verbose = True

    loss_type = 'standard_loss'

    print('DEVICE: ', device)

    if args.mode == 'train':

        train_losses = []
        val_losses = []
        test_losses = []
        epochs = []

        inverse_DNNs_from_scratch = []


        n_trials = 1
        for i in range(n_trials):

            tnn_model = tnn.tandem_model(
                            configuration=args.configuration,
                            train_data=train_data, 
                            test_data=test_data,
                            train_val_split_seed=random.randint(0,100),
                            dataset_name=args.dataset_name,
                            forward_DNN_dataset=args.forward_DNN_dataset,
                            forward_DNN_hot_start = args.forward_DNN_hot_start,
                            forward_DNN_hot_start_dataset = args.forward_DNN_hot_start_dataset,
                            inverse_DNN_hot_start_dataset=args.inverse_DNN_hot_start_dataset,
                            forward_architecture=forward_architecture, 
                            inverse_architecture=inverse_architecture,
                            num_forward_layers_transferred=args.num_forward_layers_transferred,
                            num_inverse_layers_to_transfer=args.num_inverse_layers_to_transfer,
                            epochs=max_epochs, 
                            device=device, 
                            loss_type=loss_type,
                            forward_DNN=forward_DNN,
                            verbose=verbose)   

            alpha=0
            final_train_loss, final_val_loss, epochs_to_converge = tnn_model.train(args.dataset_name, alpha=alpha)       
            emissivity_predictions, laser_parameters_predictions, test_rmse_losses, mean_loss = tnn_model.test()

            train_losses.append(final_train_loss)
            val_losses.append(final_val_loss)
            test_losses.append(mean_loss)
            epochs.append(epochs_to_converge)

            if args.configuration == 'standard':
                inverse_DNNs_from_scratch.append(tnn_model)
                best_model_idx = np.argmin(test_losses)
                best_model = inverse_DNNs_from_scratch[best_model_idx]

                torch.save(best_model.inverse_DNN.state_dict(), f'inverseDNN/{args.dataset_name}_inverse_from_scratch_forward_from_scratch.pth')
                print('Saved best inverse DNN trained from scratch with forward DNN trained from scratch')

        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)
        test_losses = np.array(test_losses)
        epochs = np.array(epochs)

        mean_train_loss = np.mean(train_losses)
        mean_val_loss = np.mean(val_losses)
        mean_test_loss = np.mean(test_losses)
        mean_epochs = np.mean(epochs)

        stdev_train_loss = np.std(train_losses)
        stdev_val_loss = np.std(val_losses)
        stdev_test_loss = np.std(test_losses)
        stdev_epochs = np.std(epochs)


        if args.dataset_name == 'inconel' or args.dataset_name == 'stainless_steel':
            result_dir = f'results/inc_ss/{loss_type}'
        elif args.dataset_name == 'airfoil_re_1_3' or args.tdataset_name == 'airfoil_re_3_6':
            result_dir = f'results/airfoil/{loss_type}'

        result_dir = f'{result_dir}/{args.num_inverse_layers_to_transfer}_layers_transferred' if args.inverse_DNN_hot_start_dataset != 'from_scratch' else result_dir
        os.makedirs(result_dir, exist_ok=True)
        forward_hot_start = 'from_scratch'
        forward_hot_start = args.forward_DNN_hot_start_dataset + 'hot_start' if args.forward_DNN_hot_start else forward_hot_start
        inverse_hot_start = 'hot_start_' + args.inverse_DNN_hot_start_dataset if args.configuration == 'transfer_learning' else 'from_scratch'
        forward_transfer_descriptor = ''
        forward_transfer_descriptor = f'{args.num_forward_layers_transferred}_layers_' if args.forward_DNN_hot_start else forward_transfer_descriptor
        outfile = f'{result_dir}/{args.configuration}_{args.dataset_name}_dataset_forwardDNN_{forward_transfer_descriptor}{forward_hot_start}_inverseDNN_{inverse_hot_start}.json'

        obj = {'mean train loss': mean_train_loss,
               'mean val loss': mean_val_loss,
               'test loss': mean_test_loss,
               'train loss std': stdev_train_loss,
               'val loss std': stdev_val_loss,
               'test loss std': stdev_test_loss,
               'epochs': mean_epochs,
               'epochs std': stdev_epochs
               }
               
        with open(outfile, 'w') as f:
            json.dump(obj, f)

    else: 
        emissivity_predictions, laser_parameters_predictions, rmse = tnn_model.test()

    #inverse_model.post_process(emissivity_predictions, laser_parameters_predictions, rmse)
    print('COMPLETE')

if __name__ == '__main__':
    main()



















