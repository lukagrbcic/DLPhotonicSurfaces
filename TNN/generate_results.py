import pandas as pd
import json

def make_results_df(dataset, hot_start_dataset, results_dir, num_inverse_layers_to_transfer, num_forward_layers_transferred):    

    df = pd.DataFrame()
    df['Config'] = ['Forward Model', 'Forward Model', 'Standard', 'Standard', 'TL - 1', 'TL - 2']
    df['Forward DNN'] = ['from_scratch', hot_start_dataset, 'from_scratch', dataset, 'from_scratch', hot_start_dataset]
    df['Inverse DNN'] = ['n/a', 'n/a', 'from_scratch', 'from_scratch', hot_start_dataset, hot_start_dataset]

    standard_fwd_from_scratch_path = f'{results_dir}/standard_{dataset}_dataset_forwardDNN_from_scratch_inverseDNN_from_scratch.json'
    standard_fwd_hot_start = f'{results_dir}/standard_{dataset}_dataset_forwardDNN_{num_forward_layers_transferred}_layers_{hot_start_dataset}_hot_start_inverseDNN_from_scratch.json'

    tl_1_path = f'{results_dir}/{num_inverse_layers_to_transfer}_layers_transferred/transfer_learning_{dataset}_dataset_forwardDNN_from_scratch_inverseDNN_hot_start_{hot_start_dataset}.json'
    tl_2_path = f'{results_dir}/{num_inverse_layers_to_transfer}_layers_transferred/transfer_learning_{dataset}_dataset_forwardDNN_{num_forward_layers_transferred}_layers_{hot_start_dataset}_hot_start_inverseDNN_hot_start_{hot_start_dataset}.json'

    

    if dataset == 'inconel':
        train_losses = [0.027089, 0.02697]
        val_losses = [0.02755, 0.02774]
        test_losses = [0.02145, 0.02201]
    elif dataset == 'stainless_steel':
        train_losses = [0.03525, 0.03316]
        val_losses = [0.03434, 0.03238]
        test_losses = [0.02893, 0.02623]


    train_std = [0.0, 0.0]
    val_std = [0.0, 0.0]
    test_std = [0.0, 0.0]

    epochs = [0.0, 0.0]
    epochs_std = [0.0, 0.0]

    paths = [standard_fwd_from_scratch_path, standard_fwd_hot_start, tl_1_path, tl_2_path]

    print(paths)

    for path in paths:

        with open(path, 'r') as f:
            data = json.load(f)

        train_losses.append(data['mean train loss'])
        val_losses.append(data['mean val loss'])
        test_losses.append(data['test loss'])
        train_std.append(data['train loss std'])
        val_std.append(data['val loss std'])
        test_std.append(data['test loss std'])
        epochs.append(data['epochs'])
        epochs_std.append(data['epochs std'])

    train_col = [f'{tl:.5f} (+/- {tl_std:.5f})' for tl, tl_std in zip(train_losses, train_std)]
    val_col = [f'{vl:.5f} (+/- {vl_std:.5f})' for vl, vl_std in zip(val_losses, val_std)]
    test_col = [f'{tl:.5f} (+/- {tl_std:.5f})' for tl, tl_std in zip(test_losses, test_std)]
    epoch_col = [f'{e} (+/- {e_std:.2f})' for e, e_std in zip(epochs, epochs_std)]

    df['Train RMSE'] = train_col
    df['Val RMSE'] = val_col
    df['Test RMSE'] = test_col
    df['Epochs'] = epoch_col

    df = df.set_index('Config')

    return df