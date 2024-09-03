import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.model_selection import train_test_split

plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})



class tandem_model:
    
    def __init__(self, train_data, test_data,
                 forward_architecture, inverse_architecture, epochs, device,
                 batch_size = 64,
                 forward_model=None, 
                 verbose=True, 
                 rmse_loss=False):
        
        self.train_data = train_data #tuple (inputs, outputs)
        self.test_data = test_data #tuple (inputs, outputs)
        self.forward_architecture = forward_architecture #forward DNN architecture
        self.inverse_architecture = inverse_architecture #inverse DNN architecutre
        self.epochs = epochs 
        self.batch_size = batch_size
        self.forward_model = forward_model #tuple (ml_model, pca_model) #load forward DNN here (include minmax scaler)
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


    @staticmethod 
    def _normalize(parameters):
        
        lspeed_min, spacing_min, power_min = 0.2, 10, 15 
        lspeed_max, spacing_max, power_max = 1.3, 700, 28
        
        lspeed_norm = (parameters[:,0] - lspeed_min)/(lspeed_max - lspeed_min)
        spacing_norm = (parameters[:, 1] - spacing_min)/(spacing_max - spacing_min)
        power_norm = (parameters[:, 2] - power_min)/(power_max - power_min)
            
        return lspeed_norm, spacing_norm, power_norm


    def get_nepd(self, preds, test_data):
        
        lspeed_norm_true, spacing_norm_true, power_norm_true = self._normalize(test_data)
        lspeed_norm_pred, spacing_norm_pred, power_norm_pred = self._normalize(preds)
    
        normalized_true = np.hstack((lspeed_norm_true, spacing_norm_true, power_norm_true))
        normalized_pred = np.hstack((lspeed_norm_pred, spacing_norm_pred, power_norm_pred))
        
        nepd_value = (1/np.sqrt(3)) * np.sqrt((lspeed_norm_true - lspeed_norm_pred)**2 +\
                                      (spacing_norm_true - spacing_norm_pred)**2 +\
                                      (power_norm_true - power_norm_pred)**2)
            
        return nepd_value
    
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
        
        X_train, X_val, y_train, y_val = train_test_split(self.train_data[0], 
                                                          self.train_data[1],
                                                          test_size=0.1, 
                                                          shuffle=False, 
                                                          random_state=23)
                        

        train_loader = self.get_torch_dataloader((X_train, y_train))
        val_loader = self.get_torch_dataloader((X_val, y_val))


        forward = self.forward_architecture
        forward.load_state_dict(torch.load(self.forward_model[0]))
        forward.eval()
        
        inverse = self.inverse_architecture

        def criterion(outputs, targets):
            return torch.sqrt(torch.mean((outputs - targets) ** 2))


        optimizer = optim.Adam(inverse.parameters(), lr=0.0004) #0.0002
        early_stopping_patience = 5
        best_loss = float('inf')
        epochs_no_improve = 0

        train_losses = []
        val_losses = []

        num_epochs = self.epochs
        for epoch in range(num_epochs):
            inverse.train()
            epoch_train_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = inverse(inputs)

                emissivity_output = forward(outputs)
           
                loss = criterion(emissivity_output, inputs)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()  
        
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            inverse.eval()
            with torch.no_grad():
                total_val_loss = 0
                for inputs, targets in val_loader:
                    outputs = inverse(inputs)
                    emissivity_output = forward(outputs)
        
                    loss = criterion(emissivity_output, inputs)       
        
                    total_val_loss += loss.item()  
                    
                avg_val_loss = total_val_loss / len(val_loader) 
                val_losses.append(avg_val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_wts = inverse.state_dict().copy()
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve == early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break


            
        torch.save(inverse.state_dict(), 'inverseModel/inverse_model.pth')

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
            
        plt.savefig('TNN_loss.pdf', bbox_inches='tight', format='pdf', dpi=500)

        
    
    def test(self):
        
        def criterion(outputs, targets):
            return torch.sqrt(torch.mean((outputs - targets) ** 2))
        
        print ('TESTING MODE')
        print ('Using:', self.device)
        
        
        forward = self.forward_architecture
        forward.load_state_dict(torch.load(self.forward_model[0]))
        forward.eval()
        
        inverse = self.inverse_architecture
        inverse.load_state_dict(torch.load('./inverseModel/inverse_model.pth'))
        inverse.eval()
        
        test_loader = self.get_torch_dataloader(self.test_data, inference=True)
        pca_model = self.forward_model[1]

        predictions = []
        laser_params = []
        rmse_loss = []
        with torch.no_grad():

            for inputs, targets in test_loader:
                
                
                outputs = inverse(inputs)
                laser_params.append(outputs.detach().cpu().numpy())                
                emissivity_output = forward(outputs)
        
                predictions.append(emissivity_output.detach().cpu().numpy())
                
                rmse = criterion(emissivity_output, inputs)
                rmse_loss.append(rmse.cpu().numpy())
            
        emissivity_predictions = np.concatenate(predictions)
        laser_params_predictions = np.concatenate(laser_params)
        rmse_loss = [i.item() for i in rmse_loss]
        return emissivity_predictions, laser_params_predictions, rmse_loss
        
    def post_process(self, emissivity_predictions, laser_params_predictions, rmse):
        
        preds = self.forward_model[1].inverse_transform(laser_params_predictions)
        nepd = self.get_nepd(preds, self.test_data[1])
        rmse = np.array(rmse)*100
       
        plt.figure(figsize=(6,5))
        plt.scatter(np.array(nepd), rmse, color='lightblue', marker='o', alpha=0.9)
       
       
        max_nepd = np.max(nepd)
        avg_nepd = np.mean(nepd)
        max_rmse = np.max(rmse)
        avg_rmse = np.mean(rmse)
       
        plt.axvline(max_nepd, color='grey', linestyle='--', linewidth=1)
        plt.axvline(avg_nepd, color='grey', linestyle='--', linewidth=1)
       
        plt.axhline(max_rmse, color='grey', linestyle='--', linewidth=1)
        plt.axhline(avg_rmse, color='grey', linestyle='--', linewidth=1)
       
        plt.text(max_nepd, plt.ylim()[1]*0.35, f'Max NEPD {max_nepd:.2f} \%', horizontalalignment='right', rotation=90)
        plt.text(avg_nepd+0.02, plt.ylim()[1]*0.35, f'Avg. NEPD {avg_nepd:.2f} \%', horizontalalignment='left', rotation=90)
       
     
        right_edge = plt.xlim()[1]
        padding = (right_edge - plt.xlim()[0]) * 0.01  # 2% padding from the right edge
        plt.text(right_edge - padding, max_rmse, f'Max RMSE {max_rmse:.2f} \%', verticalalignment='bottom', horizontalalignment='right')
        plt.text(right_edge - padding, avg_rmse-0.1, f'Avg. RMSE {avg_rmse:.2f} \%', verticalalignment='top', horizontalalignment='right')
       
       
        plt.xlabel('Design novelty (NEPD)')
        plt.ylabel('RMSE (\%)')
        plt.ylim(0, 10)
        plt.xlim(0, 1)
       
        ax = plt.gca()
       
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        plt.savefig('rmse_vs_nepd_TNN_inconel.pdf', bbox_inches='tight', format='pdf', dpi=500)
        

        

        
        





