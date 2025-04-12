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
import os
import json


import sys
sys.path.insert(0, '../..')
import DLPhotonicSurfaces.TNN.src.model.dnn as invfow
from DLPhotonicSurfaces.TNN.src.config import load_config

from src.scripts.load_data import get_paths_for_forward_training

# seed = 23
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True

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
        '--hot_start_dataset',
        type=str,
    )
    parser.add_argument(
        '--num_layers_to_transfer',
        type=int,
        default=1
    )
    parser.add_argument(
        '--hot_start_type',
        type=str
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_input_path, train_output_path, test_input_path, test_output_path = get_paths_for_forward_training(args.dataset_name)
    train_loader, val_loader, test_loader, input_size, output_size = build_dataloaders(train_input_path, train_output_path, test_input_path, test_output_path, device, args.dataset_name)

    train_losses = []
    val_losses = []
    epochs_to_converge = []
    test_losses = []

    n_trials = 20
    for i in range(n_trials):
        if args.mode == 'train':
            print('Performing training followed by inference\n')
            ### we are training a forward DNN from scratch
            if args.configuration == 'standard':
                model = invfow.forwardMLP(input_size, output_size).to(device)  
                print(model)
            
            ### we are hot starting
            elif args.configuration == 'transfer_learning':
                model = transfer_layers(args, input_size, output_size, device)

                count = 0
                for p in model.parameters():
                    if count < args.num_layers_to_transfer*2:
                        assert p.requires_grad == False
                    else:
                        assert p.requires_grad == True
                    count += 1

                for p in model.parameters():
                    print(f'Shape of weight matrix: {p.data.shape}, Requires grad: {p.requires_grad}')

                print('TRANSFER COMPLETE\n')

            config = load_config(args.config_file_path)
            train_loss, val_loss, epochs = train(model,
                                            config,
                                            train_loader,
                                            val_loader,
                                            args.dataset_name,
                                            setting=args.configuration,
                                            hot_start_dataset=args.hot_start_dataset,
                                            hot_start_type=args.hot_start_type)

            train_losses.append(train_loss[-1])
            val_losses.append(val_loss[-1])
            epochs_to_converge.append(epochs)

        else:
            print('Loading pretrained model and performing inference\n')
            model = invfow.forwardMLP(input_size, output_size).to(device)
            LOAD_PATH = f'forwardDNN/{args.dataset_name}_with_{model.num_layers_to_transfer}_layer_{args.hot_start_dataset}_{args.hot_start_type.upper()}_hot_start_{args.hot_start_type.upper()}_forward_DNN.pth' if args.configuration == 'transfer_learning' \
            else f'forwardDNN/{args.dataset_name}_forward_DNN.pth'
            model.load_state_dict(torch.load(LOAD_PATH))

        predictions, rmse_loss = inference(model, test_loader)
        test_losses.append(rmse_loss)

    mean_train_loss = np.mean(train_losses)
    mean_val_loss = np.mean(val_losses)
    mean_test_loss = np.mean(test_losses)
    mean_epochs = np.mean(epochs_to_converge)

    stdev_train_loss = np.std(train_losses)
    stdev_val_loss = np.std(val_losses)
    stdev_test_loss = np.std(test_losses)
    stdev_epochs = np.std(epochs_to_converge)

    obj = {'mean train loss': float(mean_train_loss),
               'mean val loss': float(mean_val_loss),
               'test loss': float(mean_test_loss),
               'train loss std': float(stdev_train_loss),
               'val loss std': float(stdev_val_loss),
               'test loss std': float(stdev_test_loss),
               'epochs': float(mean_epochs),
               'epochs std': float(stdev_epochs)
               }

    result_dir = 'forwardDNN_results'
    os.makedirs(result_dir, exist_ok=True)
    outfile_json = f'{result_dir}/{args.configuration}_{args.dataset_name}_dataset.json' if args.configuration == 'standard' else \
                   f'{result_dir}/{args.configuration}_{args.dataset_name}_dataset_{args.num_layers_to_transfer}_layer_{args.hot_start_dataset}_{args.hot_start_type.upper()}_hot_start.json' 
    outfile_npz = f'{result_dir}/{args.configuration}_{args.dataset_name}_dataset.npz' if args.configuration == 'standard' else \
                   f'{result_dir}/{args.configuration}_{args.dataset_name}_dataset_{args.num_layers_to_transfer}_layer_{args.hot_start_dataset}_{args.hot_start_type.upper()}_hot_start.npz' 
               
    with open(outfile_json, 'w') as f:
            json.dump(obj, f)

    np.savez(outfile_npz, train=train_losses, val=val_losses, test=test_losses, epoch=epochs)

    plot_results(train_loss, val_loss)

def build_dataloaders(train_input_path, train_output_path, test_input_path, test_output_path, device, dataset_name): 

    print('\n--------------------')
    print(f'LOADED {dataset_name} DATASET')
    print('--------------------\n')

    print(f'Using device: {device}')
    X_ = np.load(train_input_path)
    y_ = np.load(train_output_path)
    
    print('shape of input train data: ', X_.shape)
    print('shape of output train data: ', y_.shape)

    X_train_, X_val_, y_train_, y_val_ = train_test_split(X_, y_, test_size=0.2, shuffle=False, random_state=11)


    ## MinMaxScaler on data
    sc = MinMaxScaler(clip=True)
    X_train_ = sc.fit_transform(X_train_) 
    os.makedirs('forwardDNN/', exist_ok=True)
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

def transfer_layers(args, input_size, output_size, device):
    print()
    print(f'PERFORMING TRANSFER OF FIRST {args.num_layers_to_transfer} LAYERS')
    # random weights
    model = invfow.forwardMLP(input_size, output_size, args.num_layers_to_transfer).to(device)            
    print(model)
    print()

    # load hot start weights into model
    hot_start_model = invfow.forwardMLP(input_size, output_size).to(device)
    hot_start_model_path = f'src/forwardDNN/{args.hot_start_dataset}_forward_DNN.pth'
    print(f'TRANSFERING {args.hot_start_dataset} weights for {args.dataset_name} task')

    if args.hot_start_type == 'partial':
        print('DOING PARTIAL HOT START')
        hot_start_model.load_state_dict(torch.load(hot_start_model_path))

        assert args.hot_start_dataset != args.dataset_name
        # doing the layer transfer
        if args.num_layers_to_transfer == 1:
            model.linear1.load_state_dict(hot_start_model.linear1.state_dict())

        elif args.num_layers_to_transfer == 2:
            model.linear1.load_state_dict(hot_start_model.linear1.state_dict())
            model.linear2.load_state_dict(hot_start_model.linear2.state_dict())

        ### verifying transfer done properly
        i = 0
        for key in model.state_dict().keys():
            if i < args.num_layers_to_transfer*2:
                ## verifying transferred layers are the same
                assert torch.equal(model.state_dict()[key], hot_start_model.state_dict()[key])
                print(f'Parameters for {key} were transferred and match exactly')
            else:
                assert torch.equal(model.state_dict()[key], hot_start_model.state_dict()[key]) == False
                print(f'Parameters for {key} were not transferred')
            i += 1
    elif args.hot_start_type == 'full':
        print('DOING FULL HOT START')
        model.load_state_dict(torch.load(hot_start_model_path))

    count = 0
    for p in model.parameters():
        if count < args.num_layers_to_transfer*2:
            p.requires_grad = False
        count += 1

    return model


def train(model, config, train_loader, val_loader, dataset_name, setting, hot_start_dataset, hot_start_type):

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

    num_epochs = config['model_params']['n_epochs']
    epochs_to_converge = 0
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
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)       
                total_val_loss += loss.item()  # Accumulate validation loss

        avg_val_loss = total_val_loss / len(val_loader)  # Calculate average validation loss
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Training RMSE: {train_losses[-1]}, Validation RMSE: {val_losses[-1]}')
        
        # Early stopping logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        epochs_to_converge = epoch+1
        if epochs_no_improve == early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break


    ### saving mechanism
    SAVE_PATH = f'src/forwardDNN/{dataset_name}_with_{model.num_layers_to_transfer}_layer_{hot_start_dataset}_{hot_start_type.upper()}_hot_start_forward_DNN.pth' if setting == 'transfer_learning' \
        else f'forwardDNN/{dataset_name}_forward_DNN.pth'
    torch.save(model.state_dict(), SAVE_PATH)

    print('------------------')
    print('TRAINING COMPLETE')
    print('------------------')

    return train_losses, val_losses, epochs_to_converge

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
        for inputs, targets in tqdm(test_loader):
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            loss = criterion(outputs, targets)
            rmse_loss.append(loss.cpu().numpy())

    print ('Mean RMSE:', np.mean(rmse_loss))
    print ('Std RMSE:', np.std(rmse_loss))
    print ('Min RMSE:', np.min(rmse_loss))
    print ('Max RMSE:', np.max(rmse_loss))

    
    predictions = np.concatenate(predictions)

    print('------------------')
    print('INFERENCE COMPLETE')
    print('------------------')

    return predictions, np.mean(rmse_loss)

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
