import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import joblib

import generator_discriminator as gd
import inverse_forward as invfow
import warnings
import os


plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})

class generative_model:
    
    def __init__(self, train_data, test_data,
                 generator, discriminator, epochs, device, noise_dim, batch_size = 16,
                 forward_model=None, verbose=True, rmse_loss=False):
        
        self.train_data = train_data #tuple (inputs, outputs)
        self.test_data = test_data #tuple (inputs, outputs)
        self.generator = generator #generator object
        self.discriminator = discriminator #discriminator object
        self.epochs = epochs 
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.forward_model = forward_model #tuple (ml_model, pca_model)
        self.verbose = verbose
        self.device = device
        self.rmse_loss = rmse_loss


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

    
    def forward_prediction(self, fake_samples):
        
        forward = invfow.forwardMLP(np.shape(self.test_data[1])[1], np.shape(self.test_data[0])[1])
        forward.load_state_dict(torch.load(self.forward_model[0]))
        forward.eval()
        
        
        fake_samples_np = fake_samples.detach().cpu().numpy()
                
        pca_model = self.forward_model[1]
        
        pca_fake_samples = pca_model.transform(fake_samples_np)
                
        fwd_emissivity = forward(torch.tensor(pca_fake_samples, dtype=torch.float32))
        
        fwd_emissivity_tensor = torch.tensor(fwd_emissivity, dtype=torch.float32).to(self.device)
        
        return fwd_emissivity_tensor
                
        

    def train(self, alpha=0):
        
        print ('TRAINING MODE')
        print ('Using:', self.device)
        
        def criterion(outputs, targets):
            return torch.sqrt(torch.mean((outputs - targets) ** 2))
        

        
        noise_dim = self.noise_dim
        
        dataloader = self.get_torch_dataloader(self.train_data)

        

        
        # generator = self.generator

        
        # model_path = './generatorModel/generator.pth'
        # if os.path.exists(model_path):
        #     generator = gd.Generator(self.noise_dim).to(self.device)
        #     generator.load_state_dict(torch.load(model_path))
        # else:
        generator = self.generator


        discriminator = self.discriminator

   
    
        forward = invfow.forwardMLP(np.shape(self.test_data[1])[1], np.shape(self.test_data[0])[1])
        forward.load_state_dict(torch.load(self.forward_model[0]))
        forward.eval()
        scaler = self.forward_model[1]





        g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
        criterion = nn.BCELoss()

        num_epochs = self.epochs
        g_loss_plot, d_loss_plot, rmse_loss_plot = [], [], []
        
        rmse_loss = []
               
                
        for epoch in range(num_epochs):
            epoch_rmse_loss = 0.0  # Initialize epoch RMSE loss
            num_batches = 0
            for conditions, real_samples in dataloader:
                num_batches += 1  # Increment batch counter
                d_optimizer.zero_grad()
                noise = torch.randn(conditions.shape[0], noise_dim, device=self.device)
                fake_samples = generator(noise, conditions)
                
                
                """RMSE forward DNN monitor"""
                fake_samples_np = fake_samples.detach().cpu().numpy()
                lp_scaled = scaler.transform(fake_samples_np)
                emissivity_predicted = forward(torch.Tensor(lp_scaled))        
                rmse_loss = self.rmse(emissivity_predicted.to(self.device), conditions)#.item()
                epoch_rmse_loss += rmse_loss.item()
                
                
                real_labels = torch.ones(real_samples.size(0), 1, device=self.device)
                fake_labels = torch.zeros(real_samples.size(0), 1, device=self.device)
                real_loss = criterion(discriminator(conditions, real_samples), real_labels)
                fake_loss = criterion(discriminator(conditions, fake_samples.detach()), fake_labels)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                d_optimizer.step()


                g_optimizer.zero_grad()
                gen_loss = criterion(discriminator(conditions, fake_samples), real_labels) #+ rmse_loss
     
                gen_loss.backward()
                g_optimizer.step()

                g_loss_plot.append(gen_loss.item())
                d_loss_plot.append(d_loss.item())
                

            avg_rmse_loss = epoch_rmse_loss / num_batches
            print ('Epoch:', epoch, 'RMSE:', avg_rmse_loss)
            rmse_loss_plot.append(avg_rmse_loss)
        
        torch.save(generator.state_dict(), f'generatorModel/generator.pth')
        
        plt.figure(figsize=(6, 5))
        plt.plot(np.array(g_loss_plot), label='Generator Loss')
        plt.plot(np.array(d_loss_plot), label='Discriminator Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        ax = plt.gca()
               
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
            
        plt.savefig('g_d_loss.pdf', bbox_inches='tight', format='pdf', dpi=500)
   
        plt.figure(figsize=(6, 5))
        plt.plot(np.array(rmse_loss_plot)*100, label='Forward model loss')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE Loss (\%)')
        plt.ylim(0, 20)
        plt.legend()
        ax = plt.gca()
               
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
            
        plt.savefig('TNN_loss.pdf', bbox_inches='tight', format='pdf', dpi=500)
    def test(self):
        
        
        print ('TESTING MODE')
        print ('Using:', self.device)
        
        def criterion(outputs, targets):
            return torch.sqrt(torch.mean((outputs - targets) ** 2))
        
        generator = self.generator
        generator.load_state_dict(torch.load(f'generatorModel/generator1500.pth'))
        generator.eval()
        
        forward = invfow.forwardMLP(np.shape(self.test_data[1])[1], np.shape(self.test_data[0])[1])
        forward.load_state_dict(torch.load(self.forward_model[0]))
        forward.eval()

        
        noise_dim = self.noise_dim
        dataloader = self.get_torch_dataloader(self.test_data, inference=True)
        
        laser_params = []
        emissivity_predictions = []
        rmse_loss = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                            
                noise_vector = torch.randn(1, noise_dim, device=self.device)
                predictions = generator(noise_vector, inputs)
                
                lp_check = predictions.detach().cpu().numpy()[0]
                                
                lb = np.array([0.2, 10, 15])
                ub = np.array([1.3, 700, 28])
                
                while lp_check[0] > ub[0] and lp_check[0] < lb[0] and lp_check[1] > ub[1] and lp_check[1] < lb[1] and lp_check[2] > ub[2] and lp_check[2] < lb[2]:
                    predictions = generator(noise_vector, inputs)
                    lp_check = predictions.detach().cpu().numpy()


    
                pca_laser_params = self.forward_model[1].transform(predictions.detach().cpu().numpy())
                pca_laser_tensor = torch.tensor(pca_laser_params, dtype=torch.float32)#.to(self.device)
                
                
                emissivity_output = forward(pca_laser_tensor)
                
                rmse = criterion(emissivity_output.to(self.device), inputs) 
                
                rmse_loss.append(rmse.item())
                emissivity_predictions.append(emissivity_output.detach().cpu().numpy())
                laser_params.append(predictions.detach().cpu().numpy())


            
        
        
        emissivity_predictions = np.concatenate(emissivity_predictions)
        laser_params_predictions = np.concatenate(laser_params)

        
        return emissivity_predictions, laser_params_predictions, rmse_loss
    
    
    def post_process(self, emissivity_predictions, laser_params_predictions, rmse_loss):
            
      
        
        preds = laser_params_predictions
  
        lb = np.array([0.2, 10, 15])
        ub = np.array([1.3, 700, 28])
        
        preds = np.clip(preds, lb, ub)
        
        
        
        nepd = self.get_nepd(preds, self.test_data[1])
        rmse = np.array(rmse_loss)*100
       
        plt.figure(figsize=(6,5))
        plt.scatter(np.array(nepd), rmse, color='lightblue', marker='o', alpha=0.9)
       
       
        max_nepd = np.max(nepd)
        avg_nepd = np.mean(nepd)
        max_rmse = np.max(rmse)
        avg_rmse = np.mean(rmse)
       
        
        print ('max nepd:', max_nepd)
        print ('max rmse:', max_rmse)
        print ('avg nepd:', avg_nepd)
        print ('avg rmse:', avg_rmse)
        
        plt.axvline(max_nepd, color='grey', linestyle='--', linewidth=1)
        plt.axvline(avg_nepd, color='grey', linestyle='--', linewidth=1)
       
        plt.axhline(max_rmse, color='grey', linestyle='--', linewidth=1)
        plt.axhline(avg_rmse, color='grey', linestyle='--', linewidth=1)
        
        plt.ylim(0, 10)
        plt.xlim(0, 1)
        
        plt.text(max_nepd, plt.ylim()[1]*0.35, f'Max NEPD {max_nepd:.2f} \%', horizontalalignment='right', rotation=90)
        plt.text(avg_nepd+0.02, plt.ylim()[1]*0.35, f'Avg. NEPD {avg_nepd:.2f} \%', horizontalalignment='left', rotation=90)
       
     
        right_edge = plt.xlim()[1]
        # padding = (right_edge - plt.xlim()[0]) * 0.01  # 2% padding from the right edge
        # plt.text(right_edge - padding, max_rmse, f'Max RMSE {max_rmse:.2f} \%', verticalalignment='bottom', horizontalalignment='right')
        # plt.text(right_edge - padding, avg_rmse-0.1, f'Avg. RMSE {avg_rmse:.2f} \%', verticalalignment='top', horizontalalignment='right')
       
       
        plt.xlabel('Design novelty (NEPD)')
        plt.ylabel('RMSE (\%)')

       
        ax = plt.gca()
       
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        plt.savefig('rmse_vs_nepd_cGAN_inconel.pdf', bbox_inches='tight', format='pdf', dpi=500)
       
        





