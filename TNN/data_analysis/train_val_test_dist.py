import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def make_train_val_test_dist(dataset: str):

    if dataset == 'inconel':
        train_output_path = '/home/vpatro/TNN_data/inconel_data/input_train_data.npy'
        train_input_path = '/home/vpatro/TNN_data/inconel_data/output_train_data.npy'
        test_input_path = '/home/vpatro/TNN_data/inconel_data/output_test_data.npy'

    elif dataset == 'stainless_steel':
        train_output_path = '/home/vpatro/TNN_data/ss_data/input_train_data.npy'
        train_input_path = '/home/vpatro/TNN_data/ss_data/output_train_data.npy'
        test_input_path = '/home/vpatro/TNN_data/ss_data/output_test_data.npy'

    elif dataset == 'airfoil_re_1_3':
        train_output_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/input_train_data.npy'
        train_input_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/output_train_data.npy'
        test_output_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/input_test_data.npy'
        test_input_path = '/home/vpatro/TNN_data/airfoil_Re_1_3_data/output_test_data.npy'
    elif dataset == 'airfoil_re_3_6':
        train_output_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/input_train_data.npy'
        train_input_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/output_train_data.npy'
        test_output_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/input_test_data.npy'
        test_input_path = '/home/vpatro/TNN_data/airfoil_Re_3_6_data/output_test_data.npy'

    train_output = np.load(train_output_path)
    train_input = np.load(train_input_path)
    x_test = np.load(test_input_path)


    x_train, x_val, y_train, y_val = train_test_split(train_input, 
                                                            train_output,
                                                            test_size=0.1, 
                                                            shuffle=False, 
                                                            random_state=23)

    print(f'x train: {len(x_train)} samples')                      
    print(f'x val: {len(x_val)} samples')                      
    print(f'x test: {len(x_test)} samples')                                                   


    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train_plus_sig = x_train_mean + x_train_std
    x_train_minus_sig = x_train_mean - x_train_std

    x_val_mean = np.mean(x_val, axis=0)
    x_val_std = np.std(x_val, axis=0)
    x_val_plus_sig = x_val_mean + x_val_std
    x_val_minus_sig = x_val_mean - x_val_std

    x_test_mean = np.mean(x_test, axis=0)
    x_test_std = np.std(x_test, axis=0)
    x_test_plus_sig = x_test_mean + x_test_std
    x_test_minus_sig = x_test_mean - x_test_std

    plt.plot(x_train_mean, label = ' train mean', color = 'blue')
    plt.plot(x_train_plus_sig, label = 'train + sigma', color = 'blue', linestyle='dotted')
    plt.plot(x_train_minus_sig, label = 'train - sigma', color = 'blue', linestyle='dashed')

    plt.plot(x_val_mean, label = ' val mean', color = 'orange')
    plt.plot(x_val_plus_sig, label = 'val + sigma', color = 'orange', linestyle='dotted')
    plt.plot(x_val_minus_sig, label = 'val - sigma', color = 'orange', linestyle='dashed')

    plt.plot(x_test_mean, label = ' test mean', color = 'green')
    plt.plot(x_test_plus_sig, label = 'test + sigma', color = 'green', linestyle='dotted')
    plt.plot(x_test_minus_sig, label = 'test - sigma', color = 'green', linestyle='dashed')
    plt.title(f'Mean Emissivity Values - {dataset}')
    plt.legend()
    plt.show()


def inconel_ss_dist(population: str):


    inconel_train_output_path = '/home/vpatro/TNN_data/inconel_data/input_train_data.npy'
    inconel_train_input_path = '/home/vpatro/TNN_data/inconel_data/output_train_data.npy'

    ss_train_output_path = '/home/vpatro/TNN_data/ss_data/input_train_data.npy'
    ss_train_input_path = '/home/vpatro/TNN_data/ss_data/output_train_data.npy'


    inconel_test_input_path = '/home/vpatro/TNN_data/inconel_data/output_test_data.npy'
    ss_test_input_path = '/home/vpatro/TNN_data/ss_data/output_test_data.npy'

    inconel_x_test = np.load(inconel_test_input_path)
    ss_x_test = np.load(ss_test_input_path)

    inconel_train_output = np.load(inconel_train_output_path)
    inconel_train_input = np.load(inconel_train_input_path)

    inconel_x_train, inconel_x_val, inconcel_y_train, inconel_y_val = train_test_split(inconel_train_input, 
                                                        inconel_train_output,
                                                        test_size=0.1, 
                                                        shuffle=False, 
                                                        random_state=23)

    ss_x_train, ss_x_val, ss_y_train, ss_y_val = train_test_split(ss_train_input_path, 
                                                        ss_train_output_path,
                                                        test_size=0.1, 
                                                        shuffle=False, 
                                                        random_state=23)

    fig, axs = plt.subplots(1, 3)
    axs[0].hist(inconel_x_train, color='blue', bins=20, label='inconel')
    axs[0].hist(ss_x_train, color='orange', bins=20, label='ss')

    axs[1].hist(inconel_x_val, color='blue', bins=20, label='inconel')
    axs[1].hist(ss_x_val, color='orange', bins=20, label='ss')

    axs[2].hist(inconel_x_test, color='blue', bins=20, label='inconel')
    axs[2].hist(ss_x_test, color='orange', bins=20, label='ss')




                                                        