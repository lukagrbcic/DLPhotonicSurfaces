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

import argparse
import pandas as pd
import json
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'configuration',
        type=str,
        default='standard',
        help='enter whether you are performing the standard TNN configuration or Transfer Learning'
    )

    parser.add_argument(
        'dataset_name',
        type=str,
        help='enter the name of the dataset'
    )

    parser.add_argument(
        'forward_DNN_dataset',
        type=str,
        default=None,
        help='enter the name of the dataset the forward DNN was trained on'
    )

    parser.add_argument(
        '--inverse_DNN_dataset',
        type=str,
        default=None,
        help='enter the name of the dataset the inverse DNN was trained on'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default = 'train',
        help = 'enter train to do training followed by inference and enter inference to do inference on a pretrained model'
    )

    parser.add_argument(
        '--num_inverse_layers_frozen',
        type=int,
        default=0,
        help='enter how many layer of the inverse DNN should be frozen in the TL configuration'
    )

    args = parser.parse_args()

    #Since it is inverse training, reverse the data inputs
    if args.dataset_name == 'inconel':
        train_output_path = '/home/vpatro/TNN_data/inconel_data/input_train_data.npy'
        train_input_path = '/home/vpatro/TNN_data/inconel_data/output_train_data.npy'
        test_output_path = '/home/vpatro/TNN_data/inconel_data/input_test_data.npy'
        test_input_path = '/home/vpatro/TNN_data/inconel_data/output_test_data.npy'
    elif args.dataset_name == 'stainless_steel':
        train_output_path = '/home/vpatro/TNN_data/ss_data/input_train_data.npy'
        train_input_path = '/home/vpatro/TNN_data/ss_data/output_train_data.npy'
        test_output_path = '/home/vpatro/TNN_data/ss_data/input_test_data.npy'
        test_input_path = '/home/vpatro/TNN_data/ss_data/output_test_data.npy'
    elif args.dataset_name == 'airfoil_re_1_3':
        train_input_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/input_train_data.npy'
        train_output_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/output_train_data.npy'
        test_input_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/input_test_data.npy'
        test_output_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/output_test_data.npy'
    elif args.dataset_name == 'airfoil_re_3_6':
        train_input_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/input_train_data.npy'
        train_output_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/output_train_data.npy'
        test_input_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/input_test_data.npy'
        test_output_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/output_test_data.npy'


    X_train = np.load(train_input_path)
    y_train = np.load(train_output_path)

    X_test = np.load(test_input_path)
    y_test = np.load(test_output_path)

    print('')
    print('--------------------')
    print(f'LOADED {args.dataset_name} DATASET')
    print('--------------------')
    print('')

    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    print(f'Input size: {input_size} (emissivity), Output size: {output_size} (laser parameters)')
    print('')

    print(f'shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}')
    print(f'shape of X_test: {X_test.shape}, shape of y_test: {y_test.shape}')
    print('')


    forward_architecture = invfow.forwardMLP(output_size, input_size).to(device)
    inverse_architecture = invfow.inverseMLP(input_size, output_size).to(device)

    #load the pretrained forward (with minmax scaler) and inverse DNN if they're specified as arguments

    ### if we don't give a forward_DNN_dataset (ie don't want to load a pretrained forward DNN), forward_DNN will be set to None in the tnn
    
    forward_scaler_path = f'forwardDNN/{args.forward_DNN_dataset}_scaler.pkl'
    scaler = joblib.load(forward_scaler_path)
    print(f"Scaler selected is for forward_DNN trained on {args.forward_DNN_dataset}")
    forward_DNN_path = f'forwardDNN/{args.forward_DNN_dataset}_forward_DNN.pth'
    print(f"Forward DNN selected is that which was trained on {args.forward_DNN_dataset}")
    forward_DNN = (forward_DNN_path, scaler)

    # no transfer learning configuration, inverse DNN weights initialized from scratch
    if args.configuration == 'standard':
        print('')
        print('--------------------')
        print('Standard configuration -- no transfer learning')
        print('--------------------')
        print('')

        # the dataset we train on and the pretrained dataset of forward DNN should match, and inverse DNN should be trained from scratch
        assert args.dataset_name == args.forward_DNN_dataset
        assert args.inverse_DNN_dataset == None
        inverse_DNN=args.inverse_DNN_dataset

        print('Inverse DNN weights will be initialized from scratch')

        time.sleep(2)

    else: # transfer learning configuration
        ### if we don't give an inverse_DNN_dataset (ie don't want to load a pretrained inverse DNN), inverse_DNN will be set to None in the tnn

        print('')
        print('--------------------')
        print('Transfer learning configuration')
        print(f'dataset: ', args.dataset_name)
        print(f'forwardDNN dataset: ', args.forward_DNN_dataset)
        print(f'inverseDNN dataset: {args.inverse_DNN_dataset}')
        print(f'{int(args.num_inverse_layers_frozen/2)} layers of inverse DNN frozen')
        print('--------------------')
        print('')

        time.sleep(2)

        # make sure that we are actually doing transfer learning

        # configuration 1: dataset and forward DNN dataset are the SAME and inverse DNN dataset is DIFFERENT
        if args.dataset_name == args.forward_DNN_dataset:
            assert args.inverse_DNN_dataset != args.forward_DNN_dataset

        # configuration 2: dataset and forward DNN dataset are DIFFERENT and inverse DNN trained from scratch
        # configuration 3: dataset and forward DNN dataset are DIFFERERENT and forward and inverse DNN datasets are the SAME
        if args.dataset_name != args.forward_DNN_dataset:
            assert (args.inverse_DNN_dataset == None) or (args.forward_DNN_dataset == args.inverse_DNN_dataset)

        if args.inverse_DNN_dataset != None:
            inverse_DNN = f'inverseDNN/{args.inverse_DNN_dataset}_inverse_DNN.pth'
            print(f"Inverse DNN selected is that which was trained on {args.inverse_DNN_dataset}")
        else:
            inverse_DNN=args.inverse_DNN_dataset
            print('Inverse DNN weights will be initialized from scratch')

    max_epochs = 1000
    verbose = True

    loss_type = 'standard_loss'

    print('DEVICE: ', device)


    if args.mode == 'train':

        train_losses = []
        val_losses = []
        test_losses = []
        epochs = []

        n_trials = 2
        for i in range(n_trials):

            tnn_model = tnn.tandem_model(
                            train_data=train_data, 
                            test_data=test_data,
                            train_val_split_seed=random.randint(0,100),
                            forward_architecture=forward_architecture, 
                            inverse_architecture=inverse_architecture,
                            num_inverse_layers_frozen=args.num_inverse_layers_frozen,
                            epochs=max_epochs, 
                            device=device, 
                            dataset_name=args.dataset_name,
                            forward_DNN_dataset=args.forward_DNN_dataset,
                            inverse_DNN_dataset=args.inverse_DNN_dataset,
                            loss_type=loss_type,
                            forward_DNN=forward_DNN,
                            inverse_DNN_path=inverse_DNN,
                            configuration=args.configuration,
                            verbose=verbose)   

            alpha=0
            final_train_loss, final_val_loss, epochs_to_converge = tnn_model.train(args.dataset_name, alpha=alpha)       
            emissivity_predictions, laser_parameters_predictions, test_rmse_losses, mean_loss = tnn_model.test()

            train_losses.append(final_train_loss)
            val_losses.append(final_val_loss)
            test_losses.append(mean_loss)
            epochs.append(epochs_to_converge)

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
        elif args.dataset_name == 'airfoil_re_1_3' or args.dataset_name == 'airfoil_re_3_6':
            result_dir = f'results/airfoil/{loss_type}'


        inverse_DNN_dataset = args.inverse_DNN_dataset
        if args.inverse_DNN_dataset == None:
            inverse_DNN_dataset = 'from_scratch'

        result_dir = f'{result_dir}/{int(args.num_inverse_layers_frozen/2)}_layers_frozen' if args.inverse_DNN_dataset != None else result_dir
        os.makedirs(result_dir, exist_ok=True)
        outfile = f'{result_dir}/{args.configuration}_{args.dataset_name}_dataset_forwardDNN_{args.forward_DNN_dataset}_inverseDNN_{inverse_DNN_dataset}.json'

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



















