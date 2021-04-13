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
            'patience': 600, #early stopping
            'lr_decay_patience': 20, #lr reg.
            'lr_decay_gamma': 0.7,
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
            'device': 'cuda:2',
            
            'message': ''}


##############################################################
###################### Baseline Large ########################
##############################################################

@config_ingredient.named_config
def baseline_0_large():
    print("baseline_0_large model")
    config = {
        'model' : 'baseline_0_large',
        'target_bitrate': 48000,
        'bitrate_fuzz': 600,
        'num_skips': 0,
        'num_layers': 9, # 1 AE = 2110k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.02, 0.0016, 0.025, 0.0008],
        'device': 'cuda:0',
        'alpha_decrease': 0.0,
        'lr': 5e-06,
        'num_kernel': 30,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '205208',
        'optimal_factor': 1200
    }

@config_ingredient.named_config
def baseline_1_large():
    print("baseline_1_large model")
    config = {
        'model' : 'baseline_1_large',
        'target_bitrate': 48000,
        'bitrate_fuzz': 600,
        'num_skips': 0,
        'num_layers': 8, # 1 AE = 2110k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.02, 0.0016, 0.025, 0.0008],
        'device': 'cuda:1',
        'alpha_decrease': 0.0,
        'lr': 5e-06,
        'num_kernel': 36,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '101525',
        'optimal_factor': 1000
    }

@config_ingredient.named_config
def baseline_2_large():
    print("baseline_2_large model")
    config = {
        'model' : 'baseline_2_large',
        'target_bitrate': 48000,
        'bitrate_fuzz': 600,
        'num_skips': 0,
        'num_layers': 9, # 1 AE = 2110k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.02, 0.0016, 0.025, 0.0008],
        'device': 'cpu',
        'alpha_decrease': 0.0,
        'lr': 5e-06,
        'num_kernel': 36,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '844647',
        'optimal_factor': 1500
    }

@config_ingredient.named_config
def baseline_3_large():
    print("baseline_3_large model")
    config = {
        'model' : 'baseline_3_large',
        'target_bitrate': 48000,
        'bitrate_fuzz': 600,
        'num_skips': 0,
        'num_layers': 9, # 1 AE = 2110k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.02, 0.0016, 0.025, 0.0008],
        'device': 'cuda:3',
        'alpha_decrease': 0.0,
        'lr': 5e-06,
        'num_kernel': 38,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '592247',
        'optimal_factor': 1000,
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
        'num_layers': 11, # 1 AE = 211k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.003, 0.0016, 0.009, 0.0008],
        'device': 'cpu',
        'alpha_decrease': 0.0,
        'lr': 0.00005,
        'num_kernel': 24,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '142758',
        'optimal_factor': 800,
    }

@config_ingredient.named_config
def harpnet_1_large():
    print("harpnet_1_large model")
    config = {
        'model' : 'harpnet_1_large',
        'target_bitrate': 48000,
        'bitrate_fuzz': 600,
        'num_skips': 2,
        'num_layers': 11, # 1 AE = 211k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.0015, 0.0016, 0.0035, 0.0008],
        'device': 'cpu',
        'alpha_decrease': 0.0,
        'lr': 0.000005,
        'num_kernel': 24,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '268165',
        'optimal_factor': 3000,
    }

@config_ingredient.named_config
def harpnet_2_large():
    print("harpnet_2_large model")
    config = {
        'model' : 'harpnet_2_large',
        'target_bitrate': 48000,
        'bitrate_fuzz': 600,
        'num_skips': 3,
        'num_layers': 11, # 1 AE = 211k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.0007, 0.0016, 0.002, 0.0008],
        'device': 'cpu',
        'alpha_decrease': 0.0,
        'lr': 0.000005,
        'num_kernel': 24,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '122480',
        'optimal_factor': 2000
    }

@config_ingredient.named_config
def harpnet_3_large():
    print("harpnet_3_large model")
    config = {
        'model' : 'harpnet_3_large',
        'target_bitrate': 48000,
        'bitrate_fuzz': 600,
        'num_skips': 4,
        'num_layers': 11, # 1 AE = 211k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.0014, 0.0016, 0.001, 0.0008],
        'device': 'cpu',
        'alpha_decrease': 0.0,
        'lr': 0.000005,
        'num_kernel': 24,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '11789',
        'optimal_factor': 1800
    }

##############################################################
###################### Baseline Small ########################
##############################################################

@config_ingredient.named_config
def baseline_0_small():
    print("baseline_0_small model")
    config = {
        'model' : 'baseline_0_small',
        'target_bitrate': 24000,
        'bitrate_fuzz': 300,
        'num_skips': 0,
        'num_layers': 9, # 1 AE = 2110k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.02, 0.0016, 0.025, 0.0008],
        'device': 'cuda:0',
        'alpha_decrease': 0.0,
        'lr': 5e-06,
        'num_kernel': 30,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '980444',
        'optimal_factor': 1100
    }

@config_ingredient.named_config
def baseline_1_small():
    print("baseline_1_small model")
    config = {
        'model' : 'baseline_1_small',
        'target_bitrate': 24000,
        'bitrate_fuzz': 300,
        'num_skips': 0,
        'num_layers': 8, # 1 AE = 2110k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.02, 0.0016, 0.025, 0.0008],
        'device': 'cuda:1',
        'alpha_decrease': 0.0,
        'lr': 5e-06,
        'num_kernel': 36,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '209292',
        'optimal_factor': 1000
    }

@config_ingredient.named_config
def baseline_2_small():
    print("baseline_2_small model")
    config = {
        'model' : 'baseline_2_small',
        'target_bitrate': 24000,
        'bitrate_fuzz': 300,
        'num_skips': 0,
        'num_layers': 9, # 1 AE = 2110k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.02, 0.0016, 0.025, 0.0008],
        'device': 'cpu',
        'alpha_decrease': 0.0,
        'lr': 5e-06,
        'num_kernel': 36,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '770695',
        'optimal_factor': 3000
    }

@config_ingredient.named_config
def baseline_3_small():
    print("baseline_3_small model")
    config = {
        'model' : 'baseline_3_small',
        'target_bitrate': 24000,
        'bitrate_fuzz': 300,
        'num_skips': 0,
        'num_layers': 9, # 1 AE = 2110k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.02, 0.0016, 0.025, 0.0008],
        'device': 'cuda:3',
        'alpha_decrease': 0.0,
        'lr': 5e-06,
        'num_kernel': 38,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '323096',
        'optimal_factor': 3000
    }

@config_ingredient.named_config
def harpnet_0_small():
    print("harpnet_0_small model")
    config = {
        'model' : 'harpnet_0_small',
        'target_bitrate': 24000,
        'bitrate_fuzz': 300,
        'num_skips': 1,
        'num_layers': 11, # 1 AE = 211k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.02, 0.0016, 0.025, 0.0008],
        'device': 'cpu',
        'alpha_decrease': 0.0,
        'lr': 0.000009,
        'num_kernel': 24,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '116598',
        'optimal_factor': 2000
    }
 
@config_ingredient.named_config
def harpnet_1_small():
    print("harpnet_1_small model")
    config = {
        'model' : 'harpnet_1_small',
        'target_bitrate': 24000,
        'bitrate_fuzz': 300,
        'num_skips': 2,
        'num_layers': 11, # 1 AE = 211k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.015, 0.0016, 0.01, 0.0008],
        'device': 'cpu',
        'alpha_decrease': 0.0,
        'lr': 0.000009,
        'num_kernel': 24,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '544874',
        'optimal_factor': 1800
    }

@config_ingredient.named_config
def harpnet_2_small():
    print("harpnet_2_small model")
    config = {
        'model' : 'harpnet_2_small',
        'target_bitrate': 24000,
        'bitrate_fuzz': 300,
        'num_skips': 3,
        'num_layers': 11, # 1 AE = 211k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.01, 0.0016, 0.005, 0.0008],
        'device': 'cpu',
        'alpha_decrease': 0.0,
        'lr': 0.000009,
        'num_kernel': 24,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '94115',
        'optimal_factor': 1400
    } 

@config_ingredient.named_config
def harpnet_3_small():
    print("harpnet_3_small model")
    config = {
        'model' : 'harpnet_3_small',
        'target_bitrate': 24000,
        'bitrate_fuzz': 300,
        'num_skips': 4,
        'num_layers': 11, # 1 AE = 211k param
        'quant_alpha': -1,
        'loss_weights': [70.0, 1.0, 10.0], #[mse,quant,entropy]
        'tau_changes': [0.01, 0.0016, 0.005, 0.0008],
        'device': 'cpu',
        'alpha_decrease': 0.0,
        'lr': 0.000009,
        'num_kernel': 24,
        'sample_rate': 44100,
        'seq_dur': 16834,
        'root': '../../../media/sdb1/Data/ETRI_Music_LPC/',
        'model_id': '638976',
        'optimal_factor': 2500
    } 
 