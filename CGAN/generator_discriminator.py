import torch
import torch.nn as nn
import torch.optim as optim
  
    
class Generator(nn.Module):
    def __init__(self, noise_dim, input_size, output_size):
        super(Generator, self).__init__()
      
        self.model = nn.Sequential(
            nn.Linear(noise_dim + input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, noise, condition):
        combined_input = torch.cat((noise, condition), dim=1)
        combined_input = combined_input.unsqueeze(1)
        return self.model(combined_input)


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size + output_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, conditions, samples):
        x = torch.cat([conditions, samples], dim=1)
        return self.model(x)
