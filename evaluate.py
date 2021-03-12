import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"  # specify which GPU(s) to be used

from sacred import Experiment
from config import config_ingredient

import os
import torch
import csv
import numpy as np
import argparse
import soundfile as sf
import json
from pathlib import Path
import scipy.signal
import resampy
import models
import utils
import warnings
import tqdm
from contextlib import redirect_stderr
import io
import math
from matplotlib import pyplot as plt
import random

from utils import (normalize_audio)

import torchaudio

ex = Experiment('HARP Evaluation', ingredients=[config_ingredient])

@ex.config
def set_seed():
    seed = 1337

def compute_snr(x, y):
    eps = 1e-20
    ml  = np.minimum(len(x), len(y))
    return 10 * torch.log10((torch.sum(x[:ml]**2)  / torch.sum((x[:ml]-y[:ml])**2) + eps) + eps)   

@config_ingredient.capture
def load_a_model(config):

    if 'baseline' in config['model']:
        model = models.Waveunet(n_ch=config['nb_channels'], 
                                num_layers=config['num_layers'])
    elif 'harpnet' in config['model']:
        model = models.HARPNet(n_ch=config['nb_channels'], 
                                num_skips=config['num_skips'])

    for m in model.skip_encoders:
        m.quant_active = True
    
    model.quant_active = True

    model_path = os.path.join(os.path.join(config['output_dir'], config['model']), config['model_id'])
    pth_path   = next(Path(model_path).glob("%s*.pth" % config['model_id']))
    print(pth_path)

    with open(Path(model_path, config['model_id'] + '.json'), 'r') as stream:
        results = json.load(stream)

    state = torch.load(
        pth_path,
        map_location=config['device']
    )

    model.load_state_dict(state)
    model.eval()
    model.to(config['device'])

    return model

@config_ingredient.capture
def prepare_audio(config, audio):

    audio_len = audio.size()[-1]
    hop_size  = config['seq_dur'] - config['overlap']
    num_frames = math.ceil(audio_len / hop_size)

    prep_audio = torch.zeros(num_frames, config['nb_channels'], config['seq_dur'])
    timestamps = torch.zeros(num_frames,2)

    end = 0
    for i in range(num_frames):

        start = (i * hop_size)
        end   = start + config['seq_dur']

        timestamps[i,0] = start
        timestamps[i,1] = end

        if audio_len > end:
            prep_audio[i,:,:] = torch.clone(audio[:,start:end])
        else:
            last = config['seq_dur'] - (end - audio_len)
            prep_audio[i,:,:last] = torch.clone(audio[:,start:start+last])
            prep_audio[i,:,last:] = 0
    
    return prep_audio, timestamps

@config_ingredient.capture
def overlap_add(config, audio, timestamps):

    audio_ = torch.clone(audio)

    num_frames = audio_.size()[0]
    target_len = num_frames * (config['seq_dur'] - config['overlap']) + config['overlap']
    y = torch.zeros(config['nb_channels'], target_len)

    hann = torch.hann_window(config['overlap']*2, periodic=True)

    for i in range(num_frames):

        start = int(timestamps[i,0].item())
        end   = int(timestamps[i,1].item())

        chunk = torch.clone(audio_[i,:,:])
        for j in range(config['nb_channels']):
            chunk[j,:config['overlap']]  = chunk[j,:config['overlap']] * hann[:config['overlap']]
            chunk[j,-config['overlap']:] = chunk[j,-config['overlap']:] * hann[config['overlap']:]

        y[:,start:end] = y[:,start:end] + chunk

    return y
        
@config_ingredient.capture
def inference(
    config,
    audio,
    model,
):

    x, timestamps = prepare_audio(audio=audio)
    x_new = overlap_add(audio=x,timestamps=timestamps)
    x = x.to(device=config['device'])

    y_tmp = model(x)
    y_tmp_2 = y_tmp.cpu().detach().clone()

    y = overlap_add(audio=y_tmp_2,timestamps=timestamps)

    return x_new, y


def evaluate(config):

    use_cuda = torch.cuda.is_available()
    print("Using GPU:", use_cuda)

    with open('./data/test_set.csv', newline='') as f:
        reader = csv.reader(f)
        testPaths = list(reader)

    testPaths = [path for sublist in testPaths for path in sublist]
    
    model = load_a_model()

    errors = []
    count = 0
    for test_path in tqdm.tqdm(testPaths):

        count += 1
        audio = utils.load_audio(test_path, start=0, dur=None, sr=config['sample_rate'])
        if config['nb_channels'] == 1:
            audio_mono = torch.clone(torch.mean(audio, axis=0, keepdim=True))
    
        x, y = inference(model=model,audio=audio_mono)
        errors.append(compute_snr(x, y))
        # utils.soundfile_writer(os.path.join(args.main_dir,model_name+'/'+model_id+'/'+'x'+str(count)+'.wav'), x.cpu().permute(1,0).detach().numpy(), 44100)
        # utils.soundfile_writer(os.path.join(os.path.join(config['output_dir'], config['model']), config['model_id']) +'y'+str(count)+'.wav', y.cpu().permute(1,0).detach().numpy(), 44100)

    print('SNR computed:',np.mean(np.asarray(errors)))
    return np.mean(np.asarray(errors))

@ex.automain
def main(cfg):
    evaluate(config=cfg)