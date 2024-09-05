import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, noise_dim):
        self.noise_dim = noise_dim
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(822 + self.noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
        
        # self.model = nn.Sequential(
        #             nn.Linear(self.noise_dim + 822, 256),
        #             nn.LeakyReLU(0.2),
        #             nn.Linear(256, 512),
        #             nn.LeakyReLU(0.2),
        #             nn.Linear(512, 256),
        #             nn.LeakyReLU(0.2),
        #             nn.Linear(256, 128),
        #             nn.LeakyReLU(0.2),
        #             nn.Linear(128, 3),
        #             nn.Sigmoid()
        #         )
    
    # def forward(self, noise, conditions):
    #     x = torch.cat([noise, conditions], dim=1)
    #     return self.model(x)
    def forward(self, noise, conditions):
        x = torch.cat([noise, conditions], dim=1)
        raw_output = self.model(x)
        
        scaled_output = torch.zeros_like(raw_output)
        scaled_output[:, 0] = raw_output[:, 0] * 1.1 + 0.2  # Range 0.2 to 1.3
        scaled_output[:, 1] = raw_output[:, 1] * 690 + 10  # Range 10 to 700
        scaled_output[:, 2] = raw_output[:, 2] * 13 + 15  # Range 15 to 28
        
        return scaled_output

 
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(822 + 3, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, conditions, samples):
        x = torch.cat([conditions, samples], dim=1)
        return self.model(x)
    
    