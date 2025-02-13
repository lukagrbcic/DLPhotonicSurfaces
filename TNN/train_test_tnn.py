import sys
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

import inverse_forward as invfow
import tnn as tnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dataset_name',
        type=str,
        help='enter the name of the dataset'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default = 'train',
        help = 'enter train to do training followed by inference and enter inference to do inference on a pretrained model'
    )

    parser.add_argument(
        '--inverse_model_pth',
        type=str,
        help='enter the path to the inverse model to do inference on'
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
    elif args.dataset_name == 'airfoil':
        train_input_path = '/home/vpatro/TNN_data/airfoil_data/input_train_data.npy'
        train_output_path = '/home/vpatro/TNN_data/airfoil_data/output_train_data.npy'
        test_input_path = '/home/vpatro/TNN_data/airfoil_data/input_test_data.npy'
        test_output_path = '/home/vpatro/TNN_data/airfoil_data/output_test_data.npy'


    X_train = np.load(train_input_path)
    y_train = np.load(train_output_path)

    X_test = np.load(test_input_path)
    y_test = np.load(test_output_path)

    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    print('shape of X_train: ', X_train.shape)
    print('shape of y_train: ', y_train.shape)

    print('shape of X_test: ', X_test.shape)
    print('shape of y_test: ', y_test.shape)

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    print('Input size: ', input_size)
    print('Output size: ', output_size)

    #load the pretrained forward model (with minmax scaler)

    forward_scaler_path = f'forwardModel/{args.dataset_name}_scaler.pkl'
    scaler = joblib.load(forward_scaler_path)
    forward_model_path = f'forwardModel/{args.dataset_name}_forward_model.pth'
    forward_model = (forward_model_path, scaler)

    forward_architecture = invfow.forwardMLP(output_size, input_size).to(device)
    inverse_architecture = invfow.inverseMLP(input_size, output_size).to(device)

    epochs = 1000
    verbose = True

    inverse_model = tnn.tandem_model(train_data, 
                                test_data, 
                                forward_architecture, 
                                inverse_architecture, 
                                epochs, device, 
                                forward_model=forward_model,
                                verbose=verbose)   

    if args.mode == 'train':

        alpha=0
        inverse_model.train(args.dataset_name, alpha=alpha)       
        emissivity_predictions, laser_parameters_predictions, rmse = inverse_model.test(args.dataset_name)

    else:
        # load pre-trained model
        inverse_model_path = f'inverseModel/{args.dataset_name}/inverse_model.pth'
        inverse_model.load_state_dict(torch.load(inverse_model_path))
        inverse_model.train(args.dataset_name, alpha=alpha)       
        emissivity_predictions, laser_parameters_predictions, rmse = inverse_model.test(args.dataset_name)

    #inverse_model.post_process(emissivity_predictions, laser_parameters_predictions, rmse)
    print('COMPLETE')

if __name__ == '__main__':
    main()



















