from sacred import Ingredient
import numpy as np

config_ingredient = Ingredient("cfg")

@config_ingredient.config
def cfg():
    experiment_id = np.random.randint(0,1000000)
    config = {'model' : '',
            'model_id': str(experiment_id),
            'output_dir': 'output',
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
            'device': 'cpu'}

@config_ingredient.named_config
def baseline_0_large():
    print("baseline_0_large model")
    config = {
        'model' : 'baseline_0_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 0,
        'num_layers': 8, # Matching with HARP 1 AE = 110k param / 120k param
        'tau_change': 0.005,
        'quant_alpha': -20,
    }

@config_ingredient.named_config
def baseline_1_large():
    print("baseline_1_large model")
    config = {
        'model' : 'baseline_1_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 0,
        'num_layers': 10, # Matching with HARP 2 AE = 152k param / 157k param
        'tau_change': 0.005,
        'quant_alpha': -25,
    }

@config_ingredient.named_config
def baseline_2_large():
    print("baseline_2_large model")
    config = {
        'model' : 'baseline_2_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 0,
        'num_layers': 12, # Matching with HARP 3 AE = 192k param / 193k param
        'tau_change': 0.005,
        'quant_alpha': -25,
    }

@config_ingredient.named_config
def baseline_3_large():
    print("baseline_3_large model")
    config = {
        'model' : 'baseline_3_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 0,
        'num_layers': 14, # Matching with HARP 3 AE = 192k param / 193k param
        'tau_change': 0.005,
        'quant_alpha': -25,
    }

@config_ingredient.named_config
def harpnet_0_large():
    print("harpnet_0_large model")
    config = {
        'model' : 'harpnet_0_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 1,
        'num_layers': 5, # 1 AE = 110k param
        'tau_change': 0.0008,
        'quant_alpha': -40,
    }

@config_ingredient.named_config
def harpnet_1_large():
    print("harpnet_1_large model")
    config = {
        'model' : 'harpnet_1_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 2,
        'num_layers': 5, # 2 AE = 152k param
        'tau_change': 0.0008,
        'quant_alpha': -40,
    }

@config_ingredient.named_config
def harpnet_2_large():
    print("harpnet_2_large model")
    config = {
        'model' : 'harpnet_2_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 3,
        'num_layers': 5, # 3 AE = 193k param
        'tau_change': 0.0008,
        'quant_alpha': -40
    }

@config_ingredient.named_config
def harpnet_3_large():
    print("harpnet_3_large model")
    config = {
        'model' : 'harpnet_3_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 4,
        'num_layers': 5, # 4 AE = 234k param
        'tau_change': 0.0008,
        'quant_alpha': -40,
        'model_id': '598452'
    }