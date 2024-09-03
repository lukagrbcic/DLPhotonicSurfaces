import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.preprocessing import *
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib


import inverse_forward as invfow

seed = 23
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}')
X_ = np.load('../inconel_data/input_train_data.npy')
y_ = np.load('../inconel_data/output_train_data.npy')

X_train_, X_val_, y_train_, y_val_ = train_test_split(X_, y_, test_size=0.1, shuffle=False, random_state=11)


sc = MinMaxScaler(clip=True)
X_train_ = sc.fit_transform(X_train_) 
joblib.dump(sc, 'forwardModel/scaler.pkl')

X_val_ = sc.transform(X_val_)

X_test_ = np.load('../inconel_data/input_test_data.npy')
y_test_ = np.load('../inconel_data/output_test_data.npy')

X_test_ = sc.transform(X_test_)


X_train = torch.tensor(X_train_, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_, dtype=torch.float32).to(device)

X_val = torch.tensor(X_val_, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val_, dtype=torch.float32).to(device)

X_test = torch.tensor(X_test_, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

input_size = np.shape(X_)[1]
output_size = np.shape(y_)[1]

model = invfow.forwardMLP(input_size, output_size).to(device)

def criterion(outputs, targets):
    return torch.sqrt(torch.mean((outputs - targets) ** 2))


optimizer = optim.Adam(model.parameters(), lr=0.0005)
early_stopping_patience = 5
best_loss = float('inf')
epochs_no_improve = 0

train_losses = []
val_losses = []

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()  

    avg_train_loss = epoch_train_loss / len(train_loader)
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
    
torch.save(model.state_dict(), 'forwardModel/forward_model.pth')

# Define RMSE calculation function
def calculate_rmse(outputs, targets):
    return torch.sqrt(torch.mean((outputs - targets) ** 2))

# Evaluating the model
model.eval()
predictions = []
rmse_loss = []
with torch.no_grad():
    total_loss = 0
    total_rmse = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())
        loss = criterion(outputs, targets)
        rmse = calculate_rmse(outputs, targets)
        # print (rmse)
        rmse_loss.append(rmse.cpu().numpy())
        total_loss += loss.item()
        total_rmse += rmse.item()
    avg_loss = total_loss / len(test_loader)
    avg_rmse = total_rmse / len(test_loader)
    print(f'Average Test Loss: {avg_loss}')
    print(f'Average Test RMSE: {avg_rmse}')

print ('Mean RMSE:', np.mean(rmse_loss))
print ('Std RMSE:', np.std(rmse_loss))
print ('Min RMSE:', np.min(rmse_loss))
print ('Max RMSE:', np.max(rmse_loss))

    
predictions = np.concatenate(predictions)
import matplotlib.pyplot as plt

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

plt.savefig('forwardDNN_loss.pdf', bbox_inches='tight', format='pdf', dpi=500)
