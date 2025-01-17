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
        '--forward_model_path',
        type=str,
        default='forwardModel/forwardModel.pth',
        help='enter the path to the pretrained forward model'
    )

    parser.add_argument(
        '--forward_scalar_path',
        type=str,
        default='forwardModel/scaler.pkl',
        help='enter the path to the pretrained forward model scalar'
    )

    parser.add_argument(
        'inverse_config_file_path',
        type=str,
        help='enter path to config file for inverse model'
    )

    args = parser.parse_args()

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

    X_train = np.load(train_output_path)
    y_train = np.load(train_input_path)

    X_test = np.load(test_output_path)
    y_test = np.load(test_input_path)

    print('shape of X_train: ', X_train.shape)
    print('shape of y_train: ', y_train.shape)

    print('shape of X_test: ', X_test.shape)
    print('shape of y_test: ', y_test.shape)

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    model = invfow.forwardMLP(input_size, output_size).to(device)
    model.load_state_dict(torch.load(args.model_pth_path))

    #load the pretrained forward model (with minmax scaler)
    forwardDNN = './forwardModel/forward_model.pth'
    scaler = joblib.load(f'./forwardModel/scaler.pkl')
    forward_model = (forwardDNN, scaler)

    model = invfow.forwardMLP(input_size, output_size).to(device)
    model.load_state_dict(torch.load(args.forward_model_path))

results_rmse = []

#Since it is inverse training, reverse the data inputs
y_train = np.load('../inconel_data/input_train_data.npy')
X_train = np.load('../inconel_data/output_train_data.npy')

y_test = np.load('../inconel_data/input_test_data.npy')
X_test = np.load('../inconel_data/output_test_data.npy')

    
train_data = (X_train, y_train)
test_data = (X_test, y_test)


input_size = np.shape(X_train)[1]
output_size = np.shape(y_train)[1]


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

alpha=0
# inverse_model.train(alpha=alpha)       
emissivity_predictions, laser_parameters_predictions, rmse = inverse_model.test()
inverse_model.post_process(emissivity_predictions, laser_parameters_predictions, rmse)



















