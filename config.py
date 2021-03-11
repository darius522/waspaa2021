from sacred import Ingredient

config_ingredient = Ingredient("cfg")

@config_ingredient.config
experiment_id = np.random.randint(0,1000000)
def cfg():
    model_config = {'model' : '',
                    'experiment_id': str(experiment_id),
                    'root': '../../../media/sdb1/Data/ETRI_Music/',

                    # Trainig Parameters
                    'epochs': 300,
                    'num_its':1,
                    'batch_size': 16,

                    # Hyper-parameters: Quant/Entropy
                    'quant': True,
                    'quant_active': 5, # Num epoch after which the quant kicks in
                    'target_bitrate': 64000, # Target bitrate in kbps
                    'bitrate_fuzz': 450, # Allowed bitrate window
                    'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
                    'num_skips': 1, # Number of AE skip connection (only for HARP)
                    'num_layers': 8, # Number of layers to match HARP (for baseline, HARP has 5 fixed layers)
                    'tau_change': 0.005,  # change rate of the quantization regularizer loss term
                    'quant_alpha': -20, # Initial hardness of the quantization

                    'lr': 0.001,
                    'patience': 300,
                    'lr_decay_patience': 80,
                    'lr_decay_gamma': 0.3,
                    'weight_decay': 0.00001,
                    'seed': 42,

                    # Data Parameters
                    'seq_dur': 16384,
                    'overlap': 64,
                    'nb_channels': 1,
                    'sample_rate': 44100,
                    'nb_workers': 0,

                    # Misc Parameters
                    'quiet': False,
                    'device': 'cuda:0'}

@config_ingredient.named_config
def baseline_0_large():
    print("Training baseline_0_large model")
    model_config = {
        'model' : 'baseline_0_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 0,
        'num_layers': 8, # Matching with HARP 1 AE = 110k param
        'tau_change': 0.005,
        'quant_alpha': -20
    }

@config_ingredient.named_config
def harpnet_0_large():
    print("Training harpnet_0_large model")
    model_config = {
        'model' : 'baseline_0_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 1,
        'num_layers': 5, # 1 AE = 110k param
        'tau_change': 0.0008,
        'quant_alpha': -40
    }