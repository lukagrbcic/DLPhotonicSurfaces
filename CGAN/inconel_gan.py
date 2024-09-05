import sys
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

import train_generator as tg
import generator_discriminator as gd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



ml_model = f'../TNN/forwardModel/forward_model.pth'
scaler = joblib.load(f'../TNN/forwardModel/scaler.pkl')
forward_model = (ml_model, scaler)


y_train = np.load('../inconel_data/input_train_data.npy')
X_train = np.load('../inconel_data/output_train_data.npy')


y_test = np.load('../inconel_data/input_test_data.npy')
X_test = np.load('../inconel_data/output_test_data.npy')
   

train_data = (X_train, y_train)
test_data = (X_test, y_test)



noise_dim = 50

generator = gd.Generator(noise_dim).to(device)
discriminator = gd.Discriminator().to(device)
epochs = 800
verbose = True

generate = tg.generative_model(train_data, 
                            test_data, 
                            generator, 
                            discriminator, 
                            epochs, device, noise_dim=noise_dim,
                            forward_model=forward_model, 
                            verbose=verbose)   

alpha=0
# generate.train(alpha=alpha)       
emissivity_predictions, laser_params_predictions, rmse_loss = generate.test()
generate.post_process(emissivity_predictions, laser_params_predictions, rmse_loss)



















