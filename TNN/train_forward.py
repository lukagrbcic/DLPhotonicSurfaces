import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.preprocessing import *
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm
import xgboost


import sys
sys.path.insert(0, '../..')
import DLPhotonicSurfaces.TNN.dnn as invfow
from config import load_config

seed = 23
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dataset_name',
        type=str,
        help='enter the name of the dataset to be trained on'
    )
    parser.add_argument(
        'configuration',
        type=str,
        default='standard',
        help='enter whether we are doing the standard configuration (training from scratch) \
         or hot starting our forward DNN'
    )
    parser.add_argument(
        'config_file_path',
        type=str,
        help='enter path to config file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default = 'train',
        help = 'enter train to do training followed by inference and enter inference to do inference on a saved model'
    )
    parser.add_argument(
        '--hot_start_model_path',
        type=str,
    )
    parser.add_argument(
        '--num_layers_to_transfer',
        type=int,
        default=1
    )

    args = parser.parse_args()

    if args.dataset_name == 'inconel':
        train_input_path = '/home/vpatro/TNN_data/inconel_data/input_train_data.npy'
        train_output_path = '/home/vpatro/TNN_data/inconel_data/output_train_data.npy'
        test_input_path = '/home/vpatro/TNN_data/inconel_data/input_test_data.npy'
        test_output_path = '/home/vpatro/TNN_data/inconel_data/output_test_data.npy'
    elif args.dataset_name == 'stainless_steel':
        train_input_path = '/home/vpatro/TNN_data/ss_data/input_train_data.npy'
        train_output_path = '/home/vpatro/TNN_data/ss_data/output_train_data.npy'
        test_input_path = '/home/vpatro/TNN_data/ss_data/input_test_data.npy'
        test_output_path = '/home/vpatro/TNN_data/ss_data/output_test_data.npy'
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


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, input_size, output_size = load_data(train_input_path, train_output_path, test_input_path, test_output_path, device, args.dataset_name)

    if args.mode == 'train':
        print('Performing training followed by inference')
        print('\n')

        ### we are training a forward DNN from scratch
        if args.configuration == 'standard':
            model = invfow.forwardMLP(input_size, output_size).to(device)  
            hot_start_dataset = None          
            print(model.model)
        
        ### we are hot starting
        elif args.configuration == 'transfer_learning':
            print()
            print(f'PERFORMING TRANSFER OF FIRST {args.num_layers_to_transfer} LAYERS')
            # random weights
            model = invfow.forwardMLP(input_size, output_size).to(device)            
            print(model.model)
            print()

            # load hot start weights into model
            hot_start_model = invfow.forwardMLP(input_size, output_size).to(device) 
            hot_start_model.load_state_dict(torch.load(args.hot_start_model_path))

            hot_start_dataset = args.hot_start_model_path.split('/')[1].split('_')[0]
            hot_start_dataset = hot_start_dataset + '_steel' if hot_start_dataset == 'stainless' else hot_start_dataset
            print(f'TRANSFERING {hot_start_dataset} weights for {args.dataset_name} task')

            assert hot_start_dataset != args.dataset_name

            # doing the layer transfer
            if args.num_layers_to_transfer == 1:
                model.linear1.load_state_dict(hot_start_model.linear1.state_dict())

            elif args.num_layers_to_transfer == 2:
                model.linear1.load_state_dict(hot_start_model.linear1.state_dict())
                model.linear2.load_state_dict(hot_start_model.linear2.state_dict())

            count = 0
            for p in model.parameters():
                if count < args.num_layers_to_transfer*2:
                    p.requires_grad = False
                count += 1

            ### verifying that transfer done properly

            i = 0
            for key in model.state_dict().keys():
                if i < 8:
                    if i < args.num_layers_to_transfer*2:
                        ## verifying transferred layers are the same
                        assert torch.all(torch.eq(model.state_dict()[key], hot_start_model.state_dict()[key])).item()
                    else:
                        ## ReLU layer
                        if model.state_dict()[key].ndim == 1:
                            assert model.state_dict()[key][0] != hot_start_model.state_dict()[key][0]
                        ## linear layer
                        else:
                            assert model.state_dict()[key][0,0] != hot_start_model.state_dict()[key][0,0]
                else:
                    break
                i += 1

            count = 0
            for p in model.parameters():
                if count < args.num_layers_to_transfer*2:
                    assert p.requires_grad == False
                else:
                    assert p.requires_grad == True
                count += 1

            print('TRANSFER COMPLETE')
            print()

            for p in model.parameters():
                print(f'Shape of weight matrix: {p.data.shape}, Requires grad: {p.requires_grad}')


        config = load_config(args.config_file_path)
        train_losses, val_losses = train(model,
                                        config,
                                        train_loader,
                                        val_loader,
                                        args.dataset_name,
                                        setting=args.configuration,
                                        hot_start_dataset_name=hot_start_dataset)

    else:
        print('Loading pretrained model and performing inference')
        print('\n')
        model = invfow.forwardMLP(input_size, output_size).to(device)
        forward_model_path = f'forwardDNN/{args.dataset_name}_forward_DNN.pth'
        model.load_state_dict(torch.load(forward_model_path))
        config = load_config(args.config_file_path)

    predictions, rmse_losses = inference(model, test_loader)
    plot_results(train_losses, val_losses)

def load_data(train_input_path, train_output_path, test_input_path, test_output_path, device, dataset_name): 

    print('')
    print('--------------------')
    print(f'LOADED {dataset_name} DATASET')
    print('--------------------')
    print('')   

    print(f'Using device: {device}')
    X_ = np.load(train_input_path)
    y_ = np.load(train_output_path)
    
    print('shape of input train data: ', X_.shape)
    print('shape of output train data: ', y_.shape)

    X_train_, X_val_, y_train_, y_val_ = train_test_split(X_, y_, test_size=0.2, shuffle=False, random_state=11)


    ## MinMaxScaler on data
    sc = MinMaxScaler(clip=True)
    X_train_ = sc.fit_transform(X_train_) 
    joblib.dump(sc, f'forwardDNN/{dataset_name}_scaler.pkl')

    X_val_ = sc.transform(X_val_)

    X_test_ = np.load(test_input_path)
    y_test_ = np.load(test_output_path)

    print('shape of input test data: ', X_test_.shape)
    print('shape of output test data: ', y_test_.shape)

    X_test_ = sc.transform(X_test_)


    X_train = torch.tensor(X_train_, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_, dtype=torch.float32).to(device)

    X_val = torch.tensor(X_val_, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val_, dtype=torch.float32).to(device)

    X_test = torch.tensor(X_test_, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    input_size = np.shape(X_)[1]
    output_size = np.shape(y_)[1]

    return train_loader, val_loader, test_loader, input_size, output_size

def criterion(outputs, targets):
    return torch.sqrt(torch.mean((outputs - targets) ** 2))

def train(model, config, train_loader, val_loader, dataset_name, setting, hot_start_dataset_name):

    print('------------------')
    print('-----TRAINING-----')
    print('------------------')

    learning_rate = config['model_params']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping_patience = config['model_params']['early_stopping_patience']
    best_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    train_rmse_vals = []
    val_rmse_vals = []

    num_epochs = config['model_params']['n_epochs']
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()  



        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_val_rmse = 0.0
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)       
                total_val_loss += loss.item()  # Accumulate validation loss

        avg_val_loss = total_val_loss / len(val_loader)  # Calculate average validation loss
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Training RMSE: {avg_train_loss}, Validation RMSE: {avg_val_loss}')
        
        # Early stopping logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    ### saving mechanism
    if setting == 'standard':
        torch.save(model.state_dict(), f'forwardDNN/{dataset_name}_forward_DNN.pth')
    elif setting == 'transfer_learning':
        torch.save(model.state_dict(), f'forwardDNN/{dataset_name}_with_{hot_start_dataset_name}_hot_start_forward_DNN.pth')

    print('------------------')
    print('TRAINING COMPLETE')
    print('------------------')

    return train_losses, val_losses

def inference(model, test_loader):

    print('------------------')
    print('-----INFERENCE----')
    print('------------------')

    # Evaluating the model
    model.eval()
    predictions = []
    rmse_loss = []
    with torch.no_grad():
        total_loss = 0
        total_rmse = 0
        for inputs, targets in tqdm(test_loader):
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            loss = criterion(outputs, targets)
            # print (rmse)
            rmse_loss.append(loss.cpu().numpy())
            total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        print(f'Average Test Loss: {avg_loss}')

    print ('Mean RMSE:', np.mean(rmse_loss))
    print ('Std RMSE:', np.std(rmse_loss))
    print ('Min RMSE:', np.min(rmse_loss))
    print ('Max RMSE:', np.max(rmse_loss))

    
    predictions = np.concatenate(predictions)

    print('------------------')
    print('INFERENCE COMPLETE')
    print('------------------')

    return predictions, rmse_loss

def plot_results(train_losses, val_losses):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.plot(np.array(train_losses)*100, label='Training Loss')
    plt.plot(np.array(val_losses)*100, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss (\%)')
    plt.ylim(0, 10)
    plt.legend()
    ax = plt.gca()
        
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    plt.savefig('forwardDNN_loss.pdf', bbox_inches='tight', format='pdf', dpi=500)


if __name__ == '__main__':
    main()
