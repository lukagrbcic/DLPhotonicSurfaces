import sys
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

import inverse_forward as invfow
import tnn as tnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#load the pretrained forward model (with minmax scaler)
forwardDNN = './forwardModel/forward_model.pth'
scaler = joblib.load(f'./forwardModel/scaler.pkl')
forward_model = (forwardDNN, scaler)



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



















