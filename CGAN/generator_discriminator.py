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
            nn.Linear(64, 3)
        )
    
    def forward(self, noise, conditions):
        x = torch.cat([noise, conditions], dim=1)
        return self.model(x)
 
    
# class Generator(nn.Module):
#     def __init__(self, noise_dim):
#         self.noise_dim = noise_dim
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(self.noise_dim + 822, 64),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5),
#             nn.Flatten(),
#             nn.Linear(3584, 3),
#         )
    
#     def forward(self, noise, condition):
#         combined_input = torch.cat((noise, condition), dim=1)
#         combined_input = combined_input.unsqueeze(1)
#         return self.model(combined_input)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(822 + 3, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, conditions, samples):
        x = torch.cat([conditions, samples], dim=1)
        return self.model(x)
