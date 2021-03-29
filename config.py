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
            'results_csv': 'output/results.csv',

            # Trainig Parameters
            'epochs': 600,
            'num_its':1,
            'batch_size': 16,

            # Hyper-parameters: Quant/Entropy
            'quant': True,
            'quant_active': 8, # Num epoch after which the quant kicks in
            'target_bitrate': 64000, # Target bitrate in kbps
            'bitrate_fuzz': 450, # Allowed bitrate window
            'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
            'num_skips': 1, # Number of AE skip connection (only for HARP)
            'num_layers': 8, # Number of layers to match HARP (for baseline, HARP has 5 fixed layers)
            'num_kernel': 24,
            'quant_alpha': -40, # Initial hardness of the quantization
            'tau_changes': [0.005, 0.01, 0.0, 0.005], # change rate of the quantization regularizer loss term: [tau_ent_down, tau_quant_down, tau_ent_up, tau_quant_up]
            'alpha_decrease': 0.0, # alpha annealing

            'lr': 0.0001,
            'patience': 600,
            'lr_decay_patience': 50,
            'lr_decay_gamma': 0.3,
            'weight_decay': 0.000001,
            'seed': 42,

            # Data Parameters
            'seq_dur': 16384,
            'overlap': 64,
            'nb_channels': 1,
            'sample_rate': 44100,
            'nb_workers': 1,

            # Misc Parameters
            'quiet': False,
            'device': 'cuda:2'}


##############################################################
###################### Baseline Large ########################
##############################################################

@config_ingredient.named_config
def baseline_0_large():
    print("baseline_0_large model")
    config = {
        'model' : 'baseline_0_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 0,
        'num_layers': 7, # 1 AE = 110k param
        'quant_alpha': -40,
        'tau_changes': [0.0008, 0, 0.0006, 0],
        'device': 'cuda:1',
        'alpha_decrease': 0.06,
        'lr': 0.00002,
        #'model_id': '474323'
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
        'quant_alpha': -25,
        #'model_id': '484659'
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
        'quant_alpha': -25,
        #'model_id': '644376'
    }

@config_ingredient.named_config
def baseline_3_large():
    print("baseline_3_large model")
    config = {
        'model' : 'baseline_3_large',
        'target_bitrate': 64000,
        'bitrate_fuzz': 450,
        'num_skips': 0,
        'num_layers': 14, # Matching with HARP 3 AE = 234k param / 232k param
        'quant_alpha': -25,
        #'model_id': '33084'
    }

##############################################################
###################### Harpnet Large #########################
##############################################################

@config_ingredient.named_config
def harpnet_0_large():
    print("harpnet_0_large model")
    config = {
        'model' : 'harpnet_0_large',
        'target_bitrate': 48000,
        'bitrate_fuzz': 600,
        'num_skips': 1,
        'num_layers': 11, # 1 AE = 110k param
        'quant_alpha': -40,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.0008, 0.0016, 0.0001, 0.0008],
        'device': 'cuda:2',
        'alpha_decrease': 0.0,
        'lr': 0.000001,
        'num_kernel': 24,
        'sample_rate': 32000,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_32khz_LPC/',
        #'model_id': '833291'
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
        'tau_changes': [0.0008, 0.0016, 0.0, 0.0008],
        'quant_alpha': -40,
        #'model_id': '928455'
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
        'tau_changes': [0.0008, 0.0016, 0.0, 0.0008],
        'quant_alpha': -40,
        #'model_id': '168718'
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
        'tau_changes': [0.0008, 0.0016, 0.0, 0.0008],
        'quant_alpha': -40,
        #'model_id': '598452'
    }

##############################################################
###################### Baseline Small ########################
##############################################################

@config_ingredient.named_config
def baseline_0_small():
    print("baseline_0_small model")
    config = {
        'model' : 'baseline_0_small',
        'target_bitrate': 48000,
        'bitrate_fuzz': 450,
        'num_skips': 0,
        'num_layers': 5, # 1 AE = 110k param
        'quant_alpha': -20,
        'tau_changes': [0.0008, 0, 0.0006, 0],
        'device': 'cuda:2',
        'alpha_decrease': 0.03,
        'lr': 0.00002,
        #'model_id': '598238'
    }

@config_ingredient.named_config
def harpnet_0_small():
    print("harpnet_0_small model")
    config = {
        'model' : 'harpnet_0_small',
        'target_bitrate': 48000,
        'bitrate_fuzz': 600,
        'num_skips': 1,
        'num_layers': 11, # 1 AE = 110k param
        'quant_alpha': -40,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.0008, 0.0016, 0.0001, 0.0008],
        'device': 'cpu',
        'alpha_decrease': 0.0,
        'lr': 0.00001,
        #'model_id': '231080'
    }
 