import sys
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

import inverse_forward_architecture as arch
import TNN_analysis as tnn


material = 'inconel'
forwardDNN = joblib.load(f'models/{material}_model.pkl')
scaler = joblib.load(f'models/{material}_pca.pkl')
forward_model = (forwardDNN, scaler)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_rmse = []

    
train_data = (X_train, y_train)
test_data = (X_test, y_test)


input_size = np.shape(X_train)[1]
output_size = np.shape(y_train)[1]


forward_architecture = arch.forwardMLP(input_size, output_size).to(device)
inverse_architecture = arch.inverseMLP(input_size, output_size).to(device)

epochs = 500
forward_model = forward_model
verbose = True

generate = tnn.generative_model(train_data, 
                            test_data, 
                            forward_architecture, 
                            inverse_architecture, 
                            epochs, device, 
                            forward_model=forward_model, material=material, verbose=False)   

alpha=0
generate.train(alpha=alpha)       

# mean_rmse, max_rmse, std_rmse, rmse_values = generate.test(len(X_train))


        
        


# plt.figure()
# plt.plot(train_size, mean_rmse_results, 'ro-')
# # plt.plot(train_size, mean_rmse_results + std_rmse, 'r-', alpha=0.3)
# # plt.plot(train_size, mean_rmse_results - std_rmse, 'r-', alpha=0.3)
# plt.xticks(train_size)
# plt.xlabel('Training set samples')
# plt.ylabel('RMSE')
# plt.title(f'Mean RMSE, {material}, cGAN, {epochs} epochs, Test set samples: {test_size[0]}')        
# plt.savefig(f'results/cgan_{material}_mean_rmse_{epochs}.png', dpi=400)

    

# plt.figure()
# plt.plot(train_size, max_rmse_results, 'go-')
# plt.xticks(train_size)
# plt.xlabel('Training set samples')
# plt.ylabel('Max RMSE')
# plt.title(f'Max RMSE, {material}, cGAN, {epochs} epochs, Test set samples: {test_size[0]}')   
# plt.savefig(f'results/cgan_{material}_max_rmse_{epochs}.png', dpi=400)




















