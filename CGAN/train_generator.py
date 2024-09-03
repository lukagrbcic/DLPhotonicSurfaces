import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import joblib

import generator_discriminator as gd



class generative_model:
    
    def __init__(self, train_data, test_data,
                 generator, discriminator, epochs, device, material, noise_dim = 50, batch_size = 16,
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
        self.material = material
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

    
    def forward_prediction(self, fake_samples):
        
         
        fake_samples_np = fake_samples.detach().cpu().numpy()
        
        forward_model = self.forward_model[0]
        pca_model = self.forward_model[1]
        
        fwd_emissivity = pca_model.inverse_transform(forward_model.predict(fake_samples_np))
        
        fwd_emissivity_tensor = torch.tensor(fwd_emissivity, dtype=torch.float32).to(self.device)
        
        return fwd_emissivity_tensor
                
        

    def train(self, alpha=0):
        
        print ('TRAINING MODE')
        print ('Using:', self.device)
        
        print (len(self.train_data[0]))

        noise_dim = self.noise_dim
        dataloader = self.get_torch_dataloader(self.train_data)



        generator = self.generator
        discriminator = self.discriminator

        
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
        criterion = nn.BCELoss()

        num_epochs = self.epochs
        g_loss_plot, d_loss_plot, rmse_loss_plot = [], [], []

        for epoch in range(num_epochs):
            for conditions, real_samples in dataloader:

                d_optimizer.zero_grad()
                noise = torch.randn(conditions.shape[0], noise_dim, device=self.device)
                fake_samples = generator(noise, conditions)


                if self.rmse_loss == True:
                   rmse_loss = self.rmse(self.forward_prediction(fake_samples), conditions)
                
                real_labels = torch.ones(real_samples.size(0), 1, device=self.device)
                fake_labels = torch.zeros(real_samples.size(0), 1, device=self.device)
                real_loss = criterion(discriminator(conditions, real_samples), real_labels)
                fake_loss = criterion(discriminator(conditions, fake_samples.detach()), fake_labels)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                d_optimizer.step()


                g_optimizer.zero_grad()
                
                # Generator loss
                
                if self.rmse_loss == True:
                    gen_loss = criterion(discriminator(conditions, fake_samples), real_labels) + alpha*rmse_loss.item()
                
                else:
                    gen_loss = criterion(discriminator(conditions, fake_samples), real_labels)

                

                
                gen_loss.backward()
                g_optimizer.step()

                g_loss_plot.append(gen_loss.item())
                d_loss_plot.append(d_loss.item())
                
            if self.rmse_loss == True:    
                rmse_loss_plot.append(rmse_loss.item())
            
            if self.verbose == True:
                if self.rmse_loss == True:    
                    print ('RMSE/batch:', rmse_loss.item())
                    
                print(f'Epoch {epoch+1}/{num_epochs}, D Loss: {d_loss.item():.4f}, G Loss: {gen_loss.item():.4f}')
        
        torch.save(generator.state_dict(), f'generator_{self.material}/generator_{len(self.train_data[0])}.pth')
        
        plt.figure(figsize=(10, 5))
        plt.plot(g_loss_plot, label='Generator Loss')
        plt.plot(d_loss_plot, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'generator_{self.material}/generator_discriminator_loss_{len(self.train_data[0])}.png', dpi=400)
        plt.show()
        
        
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
    

        
        





