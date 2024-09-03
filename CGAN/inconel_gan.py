import sys
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

import train_generator as tg
import generator_discriminator as gd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



ml_model = joblib.load(f'../TNN/forwardModel/forward_model.pth')
pca_model = joblib.load(f'../TNN/forwardModel/scaler.pkl')
forward_model = (ml_model, pca_model)


y_train = np.load('../inconel_data/input_train_data.npy')
X_train = np.load('../inconel_data/output_train_data.npy')

y_test = np.load('../inconel_data/input_test_data.npy')
X_test = np.load('../inconel_data/output_test_data.npy')
   

train_data = (X_train, y_train)
test_data = (X_test, y_test)

noise_dim = 50
generator = gd.Generator(noise_dim, np.shape(X_train)[1], np.shape(y_train)[1]).to(device)
discriminator = gd.Discriminator(np.shape(X_train)[1], np.shape(y_train)[1]).to(device)
epochs = 500
verbose = True

generate = tg.generative_model(train_data, 
                            test_data, 
                            generator, 
                            discriminator, 
                            epochs, device, 
                            forward_model=forward_model, 
                            verbose=verbose)   

alpha=0
generate.train(alpha=alpha)       

mean_rmse, max_rmse, std_rmse, rmse_values = generate.test(len(X_train))



















