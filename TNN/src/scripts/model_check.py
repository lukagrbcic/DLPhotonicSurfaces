import joblib

def forward_DNN_check(args):

    forward_scaler_path = f'../forwardDNN/{args.forward_DNN_dataset}_scaler.pkl'
    scaler = joblib.load(forward_scaler_path)
    print(f"Scaler selected is for forward_DNN trained on {args.forward_DNN_dataset}")

    if args.forward_DNN_hot_start:
        ### either load the forward DNN that was hot started
        LOAD_PATH = f'../forwardDNN/{args.dataset_name}_with_{args.num_forward_layers_transferred}_layer_{args.forward_DNN_hot_start_dataset}_hot_start_forward_DNN.pth'
        print(f"Forward DNN was pretrained on {args.forward_DNN_dataset} with a {args.num_forward_layers_transferred} layer hot start from {args.forward_DNN_hot_start_dataset}")
    else:
        ### or use the one that was trained from scratch
        LOAD_PATH = f'../forwardDNN/{args.dataset_name}_forward_DNN.pth'
        print(f"Forward DNN was pretrained on {args.forward_DNN_dataset} with no hot start")
    print('Forward DNN frozen now')

    forward_DNN = (LOAD_PATH, scaler)
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
    assert args.inverse_DNN_hot_start_dataset != 'from_scratch'
    ## it should not be the same one we are doing the task on
    assert args.dataset_name != args.inverse_DNN_hot_start_dataset
    if args.forward_DNN_hot_start:
        assert args.forward_DNN_hot_start_dataset == args.inverse_DNN_hot_start_dataset
    else:
        assert args.forward_DNN_hot_start == False
    print(f'Inverse DNN weights were hot started with {args.inverse_DNN_hot_start_dataset}')
    print(f'{args.num_inverse_layers_to_transfer} layers of inverse DNN will be transferred and frozen')
    print('\n -------------------- \n')