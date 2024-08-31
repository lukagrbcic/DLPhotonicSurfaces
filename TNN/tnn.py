import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import joblib

import inverse_forward_architecture as ifa



class tandem_model:
    
    def __init__(self, train_data, test_data,
                 forward_architecture, inverse_architecture, epochs, device,
                 batch_size = 64,
                 forward_model=None, 
                 verbose=True, 
                 rmse_loss=False):
        
        self.train_data = train_data #tuple (inputs, outputs)
        self.test_data = test_data #tuple (inputs, outputs)
        self.forward_architecture = forward_architecture #generator object
        self.inverse_architecture = inverse_architecture #discriminator object
        self.epochs = epochs 
        self.batch_size = batch_size
        self.forward_model = forward_model #tuple (ml_model, pca_model)
        self.verbose = verbose
        self.device = device
        self.rmse_loss = rmse_loss
    
    def get_torch_dataloader(self, data, inference=False):
        
        X, y = data
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        if inference == False:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        return dataloader
    
    @staticmethod
    def rmse(outputs, targets):
        return torch.sqrt(torch.mean((outputs - targets) ** 2))

    
    def forward_prediction(self, prediction):
                 
        prediction_np = prediction.detach().cpu().numpy()
        
        forward_model = self.forward_model[0]
        pca_model = self.forward_model[1]
        
        fwd_emissivity = pca_model.inverse_transform(forward_model.predict(prediction_np))
        
        fwd_emissivity_tensor = torch.tensor(fwd_emissivity, dtype=torch.float32).to(self.device)
        
        return fwd_emissivity_tensor
                

    def train(self, alpha=0):
        
        print ('TRAINING MODE')
        print ('Using:', self.device)
        
        print (len(self.train_data[0]))

                
        dataloader = self.get_torch_dataloader(self.train_data)

        forward = self.forward_architecture
        model = self.inverse_architecture

        def criterion(outputs, targets):
            return torch.sqrt(torch.mean((outputs - targets) ** 2))


        optimizer = optim.Adam(forward_.parameters(), lr=0.0005)
        early_stopping_patience = 5
        best_loss = float('inf')
        epochs_no_improve = 0

        train_losses = []
        val_losses = []

        # Training the model
        num_epochs = self.epochs
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()  # Accumulate training loss

            avg_train_loss = epoch_train_loss / len(train_loader)  # Calculate average training loss
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

            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
            
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

            # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
            
        torch.save(model.state_dict(), 'forward_model.pth')
        
        plt.figure(figsize=(10, 5))
        plt.plot(rmse_loss_plot, label='RMSE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'generator_{self.material}/rmse_loss_{len(self.train_data[0])}.png', dpi=400)
        plt.show()
        
    
    def test(self, size):
        
        
        print ('TESTING MODE')
        print ('Using:', self.device)
        
        generator = self.generator
        generator.load_state_dict(torch.load(f'generator_{self.material}/generator_{size}.pth'))

        generator.eval()
        
        noise_dim = self.noise_dim
        dataloader = self.get_torch_dataloader(self.test_data, inference=True)
        
        rmse_values = []
        
        for inputs, targets in dataloader:

            noise_vector = torch.randn(1, noise_dim, device=self.device)
            predictions = generator(noise_vector, inputs)

            rmse_loss = self.rmse(self.forward_prediction(predictions), inputs)
            rmse_values.append(rmse_loss.item())
        
     
        std_rmse = np.std(rmse_values)
        max_rmse = np.max(rmse_values)
        mean_rmse = np.mean(rmse_values)

        
        print ('Mean RMSE:', mean_rmse)
        print ('Std RMSE:', std_rmse)
        print ('Max RMSE:', max_rmse)
        
        plt.figure()
        plt.xlabel('Value')
        plt.ylabel('RMSE')
        plt.savefig(f'generator_{self.material}/hist_rmse_loss_{size}_testing_{len(self.test_data[0])}.png', dpi=400)
        
        return mean_rmse, max_rmse, std_rmse, rmse_values
    

        
        





