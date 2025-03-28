import joblib

def forward_DNN_check(args):

    forward_scaler_path = f'forwardDNN/{args.forward_DNN_dataset}_scaler.pkl'
    scaler = joblib.load(forward_scaler_path)
    print(f"Scaler selected is for forward_DNN trained on {args.forward_DNN_dataset}")

    if args.forward_DNN_hot_start:
        ### either load the forward DNN that was hot started
        forward_DNN_path = f'forwardDNN/{args.forward_DNN_dataset}_with_{args.forward_DNN_hot_start_dataset}_hot_start_forward_DNN.pth'
        print(f"Forward DNN was pretrained on {args.forward_DNN_dataset} and hot started with {args.forward_DNN_hot_start_dataset}")
        print('Forward DNN frozen now')
    else:
        ### or use the one that was trained from scratch
        forward_DNN_path = f'forwardDNN/{args.forward_DNN_dataset}_forward_DNN.pth'
        print(f"Forward DNN was pretrained on {args.forward_DNN_dataset} with no hot start")
        print('Forward DNN frozen now')

    forward_DNN = (forward_DNN_path, scaler)
    return forward_DNN


def inverse_DNN_standard_config_check(args):
    
    assert args.dataset_name == args.forward_DNN_dataset

    print('\n -------------------- \n')
    print(f'TASK: {args.dataset_name} dataset')
    print('Standard configuration -- inverse DNN weights will be learned from scratch')
    assert args.inverse_DNN_hot_start_dataset == 'from_scratch'
    print('\n -------------------- \n')

def inverse_DNN_tl_config_check(args):

    print('\n -------------------- \n')
    print(f'TASK: {args.dataset_name} dataset')
    print('Transfer learning configuration -- we hot start the inverse DNN weights')
    ## there should be a hot start dataset for the inverse DNN
    assert args.inverse_DNN_hot_start_dataset != None
    ## it should not be the same one we are doing the task on
    assert args.dataset_name != args.inverse_DNN_hot_start_dataset
    print(f'Inverse DNN weights were hot started with {args.inverse_DNN_dataset_hot_start}')
    print(f'{int(args.num_inverse_layers_to_transfer/2)} layers of inverse DNN will be transferred and frozen')
    print('\n -------------------- \n')