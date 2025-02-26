import pandas as pd
import json

def make_results_df(dataset, transfer_dataset, results_dir):

    df = pd.DataFrame()

    df['Config'] = ['Forward Model', 'TNN Standard', 'TL - 1', 'TL - 2', 'TL - 3']
    df['Forward DNN'] = [dataset, dataset, dataset, transfer_dataset, transfer_dataset]
    df['Inverse DNN'] = ['n/a', 'from scratch', transfer_dataset, 'from scratch', transfer_dataset]

    standard_path = f'{results_dir}/standard_{dataset}_dataset_forwardDNN_{dataset}_inverseDNN_from_scratch.json'
    tl_1_path = f'{results_dir}/transfer_learning_{dataset}_dataset_forwardDNN_{dataset}_inverseDNN_{transfer_dataset}.json'
    tl2_path = f'{results_dir}/transfer_learning_{dataset}_dataset_forwardDNN_{transfer_dataset}_inverseDNN_from_scratch.json'
    tl3_path = f'{results_dir}/transfer_learning_{dataset}_dataset_forwardDNN_{transfer_dataset}_inverseDNN_{transfer_dataset}.json'

    if dataset == 'inconel':
        train_losses = [0.03099]
        val_losses = [0.03263]
        test_losses = [0.02565]
    elif dataset == 'stainless_steel':
        train_losses = [0.03392]
        val_losses = [0.03240]
        test_losses = [0.02726]


    train_std = [0.0]
    val_std = [0.0]
    test_std = [0.0]

    epochs = [0.0]
    epochs_std = [0.0]

    paths = [standard_path, tl_1_path, tl2_path, tl3_path]

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
    epoch_col = [f'{e} (+/- {e_std})' for e, e_std in zip(epochs, epochs_std)]

    df['Train RMSE'] = train_col
    df['Val RMSE'] = val_col
    df['Test RMSE'] = test_col
    df['Epochs'] = epoch_col

    df = df.set_index('Config')

    return df