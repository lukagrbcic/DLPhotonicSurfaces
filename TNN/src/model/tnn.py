import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})

   
class tandem_model():
    
    def __init__(self, 
                configuration,
                train_data,
                test_data,
                dataset_name,
                train_val_split_seed,
                forward_DNN_dataset,
                forward_DNN_hot_start,
                forward_DNN_hot_start_dataset,
                inverse_DNN_hot_start_dataset,
                forward_architecture,
                inverse_architecture,
                num_forward_layers_transferred,
                num_inverse_layers_to_transfer,
                epochs,
                device,
                loss_type,
                batch_size = 64,
                forward_DNN=None,
                verbose=True, 
                rmse_loss=False,
                ):
        
        self.train_data = train_data #tuple (inputs, outputs)
        self.test_data = test_data #tuple (inputs, outputs)
        self.train_val_split_seed = train_val_split_seed
        self.dataset_name = dataset_name
        self.forward_DNN_dataset = forward_DNN_dataset
        self.forward_DNN_hot_start = forward_DNN_hot_start
        self.forward_DNN_hot_start_dataset = forward_DNN_hot_start_dataset
        self.inverse_DNN_hot_start_dataset = inverse_DNN_hot_start_dataset
        self.forward_architecture = forward_architecture #forward DNN architecture
        self.inverse_architecture = inverse_architecture #inverse DNN architecutre
        self.num_forward_layers_transferred = num_forward_layers_transferred
        self.num_inverse_layers_to_transfer = num_inverse_layers_to_transfer
        self.epochs = epochs 
        self.dataset_name = dataset_name
        
        self.batch_size = batch_size
        self.forward_DNN = forward_DNN #tuple (ml_model, pca_model) #load forward DNN here (include minmax scaler)
        self.verbose = verbose
        self.configuration = configuration
        self.loss_type = loss_type
        self.device = device
        self.rmse_loss = rmse_loss

        self.forward = None
        self.inverse = None
    
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

    def train(self, dataset_name, alpha=0):

        def criterion(emissivity_preds, emissivity_targets,
            parameter_preds=None, parameter_targets=None, lambda_val=0.8, loss='standard_loss'):
            if loss == 'standard_loss':
                out = torch.sqrt(torch.mean((emissivity_preds - emissivity_targets) ** 2))
                return out
            else:
                emissivity_term = torch.sqrt(torch.mean((emissivity_preds - emissivity_targets) ** 2))
                laser_param_term = torch.sqrt(torch.mean((parameter_preds - parameter_targets) ** 2))

            return laser_param_term + lambda_val*emissivity_term
        
        print('\n ------------------')
        print('-----TRAINING-----')
        print('------------------')
        print (f'Using: {self.device} \n')
        
        X_train, X_val, y_train, y_val = train_test_split(self.train_data[0], 
                                                          self.train_data[1],
                                                          test_size=0.2, 
                                                          shuffle=False, 
                                                          random_state=self.train_val_split_seed)
                        

        train_loader = self.get_torch_dataloader((X_train, y_train))
        val_loader = self.get_torch_dataloader((X_val, y_val))

        ### load the forward DNN 


        ##############################
        ###### SETTING UP FORWARD DNN
        ##############################
        forward = self.forward_architecture
        forward.load_state_dict(torch.load(self.forward_DNN[0]))

        if self.forward_DNN_hot_start:
            print(f'Loading forward_DNN pretrained on {self.forward_DNN_dataset} with a hot start on {self.forward_DNN_hot_start_dataset}')
        else:
            print(f'Loading forward_DNN pretrained on {self.forward_DNN_dataset} with no hot start')

        forward.eval()
        ### deactivating gradients for forward DNN so it's not experiencing backprop
        for param in forward.parameters():
            param.requires_grad = False

        print('Forward DNN frozen')
        self.forward = forward


        ##############################
        ###### SETTING UP INVERSE DNN
        ##############################
        
        # load the inverse model, which will have randomly initialized weights to begin with
        input_size = self.train_data[0].shape[1]
        output_size = self.train_data[1].shape[1]


        if self.configuration == 'transfer_learning':
            ## we are doing transfer learning
            print('\n---- TRANSFER LEARNING CONFIGURATION ----')
            print(f'TRANSFERING {self.inverse_DNN_hot_start_dataset} weights for {self.dataset_name} task')
            
            # loading the entire pretrained model to prepare for selective weight transfer
            ## in particular, when we pull an inverse DNN to hot start, it should be a model
            ## that was trained from scratch by a forward DNN that was trained from scratch
            hot_start_model_path = f'../inverseDNN/{dataset_name}_inverse_from_scratch_forward_from_scratch.pth'

            pretrained_model = self.inverse_architecture.__class__(input_size, output_size)
            pretrained_model.load_state_dict(torch.load(hot_start_model_path))
            inverse = self.inverse_architecture.__class__(input_size, output_size)

            for k, v in inverse.state_dict().items():
                assert torch.equal(inverse.state_dict()[k], pretrained_model.state_dict()[k]) == False

            print('New inverse DNN was instantiated and contains random weights\n')

            # doing the layer transfer
            if self.num_inverse_layers_to_transfer == 1:
                inverse.linear1.load_state_dict(pretrained_model.linear1.state_dict())

            elif self.num_inverse_layers_to_transfer == 2:
                inverse.linear1.load_state_dict(pretrained_model.linear1.state_dict())
                inverse.linear2.load_state_dict(pretrained_model.linear2.state_dict())

            ### disabling gradient calculation to freeze selective layers
            count = 0
            for p in inverse.parameters():
                if count < self.num_inverse_layers_to_transfer*2:
                    p.requires_grad = False
                count += 1

            ### verifying that transfer done properly
            i = 0
            for key in inverse.state_dict().keys():
                # for the weight and the bias
                if i < self.num_inverse_layers_to_transfer*2:
                    assert torch.all(torch.eq(inverse.state_dict()[key], pretrained_model.state_dict()[key])).item()
                    print(f'Parameters for {key} were transferred and match exactly')
                else:
                    assert torch.equal(inverse.state_dict()[key], pretrained_model.state_dict()[key]) == False
                    print(f'Parameters for {key} were not transferred')
                i+=1

            print()
            print(f'FREEZING {self.num_inverse_layers_to_transfer} LAYERS')

            count = 0
            for p in inverse.parameters():
                if count < self.num_inverse_layers_to_transfer*2:
                    assert p.requires_grad == False
                else:
                    assert p.requires_grad == True
                count += 1

            print('---- TRANSFER COMPLETE ----')
            print()

            for p in inverse.parameters():
                print(f'Shape of weight matrix: {p.data.shape}, Requires grad: {p.requires_grad}')
            print()

        elif self.configuration == 'standard':
            inverse = self.inverse_architecture.__class__(input_size, output_size)
            # torch.save(inverse.state_dict().copy(), 'inverse_DNN_starting_wts.pth')
            starting_weights_path = '../inverse_DNN_starting_wts.pth'
            inverse.load_state_dict(torch.load(starting_weights_path))
            print('Initializing inverse_DNN from scratch')

        inverse.to(self.device)
        
        optimizer = optim.Adam(inverse.parameters(), lr=0.0004) #0.0002
        early_stopping_patience = 5
        best_loss = float('inf')
        epochs_no_improve = 0

        train_losses = []

        num_epochs = self.epochs
        epochs_to_converge = 0
        for epoch in range(num_epochs):
            inverse.train()
            epoch_train_loss = 0
            for train_emis_inputs, train_param_targets in tqdm(train_loader):
                optimizer.zero_grad()

    
                train_param_outputs = inverse(train_emis_inputs) # map emissivity curves to laser parameters
                train_emissivity_output = forward(train_param_outputs) # map the laser parameters back to emissivity curves

                loss = criterion(emissivity_preds=train_emissivity_output,
                                    emissivity_targets=train_emis_inputs,
                                    parameter_preds=train_param_outputs,
                                    parameter_targets=train_param_targets, lambda_val=0.8, loss=self.loss_type)

                loss.backward() # this will only change the weights of the inverse DNN
                optimizer.step()
                epoch_train_loss += loss.item()  
        
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            val_losses = []
            final_val_loss = 0.0

            inverse.eval()
            with torch.no_grad():
                total_val_loss = 0
                for val_emis_inputs, val_param_targets in val_loader:

                    # map emissivity -> laser parameters -> emissivity again
                    val_param_outputs = inverse(val_emis_inputs)
                    val_emissivity_output = forward(val_param_outputs)
        
                    loss = criterion(emissivity_preds=val_emissivity_output,
                                    emissivity_targets=val_emis_inputs,
                                    parameter_preds=val_param_outputs,
                                    parameter_targets=val_param_targets, lambda_val=0.8, loss=self.loss_type)   
        
                    total_val_loss += loss.item()  
                    
                avg_val_loss = total_val_loss / len(val_loader) 
                val_losses.append(avg_val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
            epochs_to_converge = epoch+1

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_wts = inverse.state_dict().copy()
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve == early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        ##################
        ### saving mechanism
        ##################

        forward_descriptor = 'from_scratch'
        forward_descriptor = f'{self.num_forward_layers_transferred}_layer_{self.forward_DNN_hot_start_dataset}_hot_start' if self.forward_DNN_hot_start else forward_descriptor
        if self.configuration == 'transfer_learning':
            os.makedirs(f'../../transfer_learning_models/{self.dataset_name}', exist_ok=True)
            path = f'../../transfer_learning_models/{self.dataset_name}/inverse_{self.num_inverse_layers_to_transfer}_layer_hot_start_{self.inverse_DNN_hot_start_dataset}_forward_{forward_descriptor}.pth'
        elif self.configuration == 'standard':
            os.makedirs('../inverseDNN/', exist_ok=True)
            path = f'../inverseDNN/{dataset_name}_inverse_from_scratch_forward_{forward_descriptor}.pth'
        torch.save(inverse.state_dict(), path)

        self.inverse_DNN = inverse
        print()
        print(f'Saved model to {path}')


        print('------------------')
        print('TRAINING COMPLETE')
        print('------------------')

        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]

        return final_train_loss, final_val_loss, epochs_to_converge

    
    def test(self):

        print('')
        print('------------------')
        print('-----INFERENCE----')
        print('------------------')
        print('')
        
        print ('TESTING MODE')
        print ('Using:', self.device)
        print('')

        def criterion(emissivity_preds, emissivity_targets,
            parameter_preds=None, parameter_targets=None, lambda_val=0.8, loss='standard_loss'):
            if loss == 'standard_loss':
                out = torch.sqrt(torch.mean((emissivity_preds - emissivity_targets) ** 2))
                return out
            else:
                emissivity_term = torch.sqrt(torch.mean((emissivity_preds - emissivity_targets) ** 2))
                laser_param_term = torch.sqrt(torch.mean((parameter_preds - parameter_targets) ** 2))

            return laser_param_term + lambda_val*emissivity_term

        forward = self.forward_architecture
        inverse = self.inverse_architecture

        forward.load_state_dict(torch.load(self.forward_DNN[0]))

        
        if self.forward_DNN_hot_start:
            print(f'Loading forward_DNN pretrained on {self.forward_DNN_dataset} with a hot start on {self.forward_DNN_hot_start_dataset}')
        else:
            print(f'Loading forward_DNN pretrained on {self.forward_DNN_dataset} with no hot start')
             

        forward.eval()
        ### deactivating gradients for forward DNN so it's not experiencing backprop
        for param in forward.parameters():
            param.requires_grad = False

        print('Forward DNN frozen')

        

        forward_descriptor = 'from_scratch'
        forward_descriptor = f'{self.num_forward_layers_transferred}_layer_{self.forward_DNN_hot_start_dataset}_hot_start' if self.forward_DNN_hot_start else forward_descriptor
        if self.configuration == 'transfer_learning':
            print(f'\nTransfer learning -- loading inverse DNN with hot start on {self.inverse_DNN_hot_start_dataset}')
            inverse_path = f'../../transfer_learning_models/{self.dataset_name}/inverse_{self.num_inverse_layers_to_transfer}_layer_hot_start_{self.inverse_DNN_hot_start_dataset}_forward_{forward_descriptor}.pth'
        else: # standard configuration
            os.makedirs('inverseDNN/', exist_ok=True)
            inverse_path = f'../inverseDNN/{self.dataset_name}_inverse_from_scratch_forward_{forward_descriptor}.pth'
        inverse.load_state_dict(torch.load(inverse_path))

        forward.eval()
        inverse.eval()
        
        test_loader = self.get_torch_dataloader(self.test_data, inference=True)
        pca_model = self.forward_DNN[1]

        predictions = []
        laser_params = []
        rmse_losses = []
        with torch.no_grad():

            for emis_inputs, param_targets in test_loader:
                
                param_outputs = inverse(emis_inputs)
                laser_params.append(param_outputs.detach().cpu().numpy())                
                emissivity_output = forward(param_outputs)
        
                predictions.append(emissivity_output.detach().cpu().numpy())
                
                rmse = criterion(emissivity_preds=emissivity_output,
                                    emissivity_targets=emis_inputs,
                                    parameter_preds=param_outputs,
                                    parameter_targets=param_targets, lambda_val=0.8, loss=self.loss_type)
                rmse_losses.append(rmse.cpu().numpy())
            
        emissivity_predictions = np.concatenate(predictions)
        laser_params_predictions = np.concatenate(laser_params)
        rmse_losses = [i.item() for i in rmse_losses]

        print ('Mean RMSE:', np.mean(rmse_losses))
        print ('Std RMSE:', np.std(rmse_losses))
        print ('Min RMSE:', np.min(rmse_losses))
        print ('Max RMSE:', np.max(rmse_losses))
            

        print('------------------')
        print('INFERENCE COMPLETE')
        print('------------------')

        return emissivity_predictions, laser_params_predictions, rmse_losses, np.mean(rmse_losses)
        
    def post_process(self, emissivity_predictions, laser_params_predictions, rmse):
        
        preds = self.forward_DNN[1].inverse_transform(laser_params_predictions)
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
       
        plt.text(max_nepd, plt.ylim()[1]*0.35, f'Max NEPD {max_nepd:.2f}', horizontalalignment='right', rotation=90)
        plt.text(avg_nepd+0.02, plt.ylim()[1]*0.35, f'Avg. NEPD {avg_nepd:.2f}', horizontalalignment='left', rotation=90)
       
     
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

    

        

        
        





