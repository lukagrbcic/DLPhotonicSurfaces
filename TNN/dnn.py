import torch
import torch.nn as nn
import torch.optim as optim
    
    
class forwardMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(forwardMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

        
class inverseMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(inverseMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)



class airfoil_forward_DNN(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.Tanh(),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.Tanh(),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.Tanh(),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.Tanh(),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.Tanh(),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
            return self.model(x)


class airfoil_inverse_DNN(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
            return self.model(x)
