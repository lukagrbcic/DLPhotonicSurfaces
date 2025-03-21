import numpy as np

def load_data(dataset_name):

    if dataset_name == 'inconel':
        train_output_path = '/home/vpatro/TNN_data/inconel_data/input_train_data.npy'
        train_input_path = '/home/vpatro/TNN_data/inconel_data/output_train_data.npy'
        test_output_path = '/home/vpatro/TNN_data/inconel_data/input_test_data.npy'
        test_input_path = '/home/vpatro/TNN_data/inconel_data/output_test_data.npy'
    elif dataset_name == 'stainless_steel':
        train_output_path = '/home/vpatro/TNN_data/ss_data/input_train_data.npy'
        train_input_path = '/home/vpatro/TNN_data/ss_data/output_train_data.npy'
        test_output_path = '/home/vpatro/TNN_data/ss_data/input_test_data.npy'
        test_input_path = '/home/vpatro/TNN_data/ss_data/output_test_data.npy'
    elif dataset_name == 'airfoil_re_1_3':
        train_input_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/input_train_data.npy'
        train_output_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/output_train_data.npy'
        test_input_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/input_test_data.npy'
        test_output_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/output_test_data.npy'
    elif dataset_name == 'airfoil_re_3_6':
        train_input_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/input_train_data.npy'
        train_output_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/output_train_data.npy'
        test_input_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/input_test_data.npy'
        test_output_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/output_test_data.npy'


    X_train = np.load(train_input_path)
    y_train = np.load(train_output_path)

    X_test = np.load(test_input_path)
    y_test = np.load(test_output_path)

    return X_train, y_train, X_test, y_test