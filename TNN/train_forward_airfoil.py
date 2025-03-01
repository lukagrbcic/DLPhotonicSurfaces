import numpy as np
import argparse
import joblib
import torch
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dataset_name',
        type=str,
        help='enter the name of the dataset to be processed'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default = 'train',
        help = 'enter train to do training followed by inference and enter inference to do inference on a pretrained model'
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset_name == 'airfoil_re_1_3':
        train_input_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/input_train_data.npy'
        train_output_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/output_train_data.npy'
        test_input_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/input_test_data.npy'
        test_output_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/output_test_data.npy'
    elif args.dataset_name == 'airfoil_re_3_6':
        train_input_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/input_train_data.npy'
        train_output_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/output_train_data.npy'
        test_input_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/input_test_data.npy'
        test_output_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/output_test_data.npy'

    X_train, y_train, X_test, y_test = load_data(train_input_path=train_input_path,
                                                train_output_path=train_output_path,
                                                test_input_path=test_input_path,
                                                test_output_path=test_output_path,
                                                device=device,
                                                dataset_name=args.dataset_name)

    model = xgb.XGBRegressor(n_estimators=2000, max_depth=3, eta=0.1)

    if args.mode == 'train':
        print('------------')
        print('TRAINING BEGINNING')
        print('------------')
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        print('------------')
        print('TRAINING COMPLETE')
        print('------------')

        mse = mean_squared_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mse)
    
    elif args.mode == 'inference':
        print('------------')
        print('LOADING PRETRAINED MODEL')
        print('------------')
        model_path = f'forwardDNN/{args.dataset_name}_forward_DNN.pkl'
        model = joblib.load(model_path)
        
    print('------------')
    print('INFERENCE BEGINNING')
    print('------------')
    y_pred = model.predict(X_test)
    print('------------')
    print('INFERENCE COMPLETE')
    print('------------')
    mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(mse)

    print('Train MSE: ', train_rmse)
    print('Test MSE: ', test_rmse)

    print('COMPLETE')





def load_data(train_input_path, train_output_path, test_input_path, test_output_path, device, dataset_name): 

    print('')
    print('--------------------')
    print(f'LOADED {dataset_name} DATASET')
    print('--------------------')
    print('')   

    print(f'Using device: {device}')
    X_train = np.load(train_input_path)
    y_train = np.load(train_output_path)
    
    print('shape of input train data: ', X_train.shape)
    print('shape of output train data: ', y_train.shape)

    ## MinMaxScaler on data
    sc = MinMaxScaler(clip=True)
    X_train = sc.fit_transform(X_train) 
    joblib.dump(sc, f'forwardDNN/{dataset_name}_scaler.pkl')

    X_test = np.load(test_input_path)
    y_test = np.load(test_output_path)

    print('shape of input test data: ', X_test.shape)
    print('shape of output test data: ', y_test.shape)

    X_test_ = sc.transform(X_test)

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    main()