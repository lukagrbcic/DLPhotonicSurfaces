import sys
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

import DLPhotonicSurfaces.TNN.dnn as invfow
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
        '--forward_DNN_dataset',
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

    forward_architecture = invfow.forwardMLP(output_size, input_size).to(device)
    inverse_architecture = invfow.inverseMLP(input_size, output_size).to(device)

    #load the pretrained forward (with minmax scaler) and inverse DNN if they're specified as arguments

    if args.forward_DNN_dataset != None:
        forward_scaler_path = f'forwardDNN/{args.forward_DNN_dataset}_scaler.pkl'
        scaler = joblib.load(forward_scaler_path)
        print(f"Scaler selected is for forward_DNN trained on {args.forward_DNN_dataset}")
        forward_DNN_path = f'forwardDNN/{args.forward_DNN_dataset}_forward_DNN.pkl'
        print(f"Forward DNN selected is that which was trained on {args.forward_DNN_dataset}")
        forward_DNN = (forward_DNN_path, scaler)

    if args.inverse_DNN_dataset != None:
        inverse_DNN = f'invserseDNN/{args.inverse_DNN_dataset}_forward_DNN.pth'
        print(f"Inverse DNN selected is that which was trained on {args.inverse_DNN_dataset}")

    epochs = 1000
    verbose = True

    tnn_model = tnn.tandem_model(train_data, 
                                test_data, 
                                forward_architecture, 
                                inverse_architecture, 
                                epochs, device, 
                                forward_model=forward_DNN,
                                verbose=verbose)   

    if args.mode == 'train':

        alpha=0
        tnn_model.train(args.dataset_name, alpha=alpha)       
        emissivity_predictions, laser_parameters_predictions, rmse = tnn_model.test(args.dataset_name)

    else:
        # load pre-trained model
        inverse_model_path = f'inverseModel/{args.dataset_name}/inverse_model.pth'
        tnn_model.load_state_dict(torch.load(inverse_model_path))
        tnn_model.train(args.dataset_name, alpha=alpha)       
        emissivity_predictions, laser_parameters_predictions, rmse = tnn_model.test(args.dataset_name)

    #inverse_model.post_process(emissivity_predictions, laser_parameters_predictions, rmse)
    print('COMPLETE')

if __name__ == '__main__':
    main()



















