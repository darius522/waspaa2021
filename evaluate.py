import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"  # specify which GPU(s) to be used

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
import pandas as pd
from pathlib import Path

from utils import (normalize_audio)

import torchaudio
import museval

ex = Experiment('HARP Evaluation', ingredients=[config_ingredient])

@ex.config
def set_seed():
    seed = 1337

def compute_snr(x, y):
    eps = 1e-20
    return 10 * np.log10((np.sum(x**2)  / np.sum((x-y)**2) + eps) + eps)

@config_ingredient.capture
def load_a_model(config):

    if 'baseline' in config['model']:
        model = models.Waveunet(n_ch=config['nb_channels'], 
                                num_layers=config['num_layers'],
                                H=config['seq_dur'],  
                                W=config['num_kernel'])
    elif 'harpnet' in config['model']:
        model = models.HARPNet(n_ch=config['nb_channels'],
                                num_layers=config['num_layers'],
                                num_skips=config['num_skips'],
                                H=config['seq_dur'],  
                                W=config['num_kernel'])

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

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, num_param, model_path

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


def compute_lpc(residual, coef_path):

    hann = scipy.signal.windows.hann(1024, sym=False)[:, None] ** 0.5
    A = np.load(coef_path,allow_pickle=True)
    R = utils.vec2mat(residual,1024,512,hann)
    Yh = np.zeros_like(R)
    unstableCnt=0
    for i in range(A.shape[1]):
        Yh[:,i], unstable = utils.LPC_synthesis(A[:,i], R[:,i])
        if unstable:
            unstableCnt+=1
    yh = utils.mat2vec(Yh, 1024, 512,hann)

    return yh
        
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

    test_csv = './data/test_set44_lpc.csv' if config['sample_rate'] == 44100 else './data/test_set32_lpc.csv'
    with open(test_csv, newline='') as f:
        reader = csv.reader(f)
        testPaths = list(reader)

    output_path = os.path.join(os.path.join(os.path.join(config['output_dir'], config['model']), config['model_id']),'songs')
    if not os.path.isdir(output_path): os.mkdir(output_path)

    testPaths = [path for sublist in testPaths for path in sublist]
    
    model, num_param, model_path = load_a_model()

    #toWrite = ['1710-00001-PFO','1136-00001-JNO','1265-00003-PFO']

    factor =config['optimal_factor']
    snrs = []
    sdrs = []
    count = 0
    print('Computing for Factor:',factor)
    for test_path in tqdm.tqdm(testPaths):

        count += 1

        audio = utils.load_audio(test_path, start=0, dur=None, sr=config['sample_rate'])

        if config['nb_channels'] == 1:
            audio = torch.clone(torch.mean(audio, axis=0, keepdim=True))

        audio  *= factor
    
        x, y = inference(model=model,audio=audio)

        x /= factor
        y /= factor

        x_lpc = compute_lpc(np.squeeze(x.cpu().permute(1,0).detach().numpy()),test_path.replace('_residual.wav','_Coef.npy'))
        y_lpc = compute_lpc(np.squeeze(y.cpu().permute(1,0).detach().numpy()),test_path.replace('_residual.wav','_Coef.npy'))

        ml  = np.minimum(len(x_lpc), len(y_lpc))
        snrs.append(compute_snr(x_lpc[:ml], y_lpc[:ml]))
        #sdrs.append(np.mean(museval.evaluate(x_lpc[:ml][np.newaxis,...], y_lpc[:ml][np.newaxis,...])[0]))

        #utils.soundfile_writer(os.path.join(os.path.join(config['output_dir'], config['model']), config['model_id']) +'/x'+str(count)+'_ref.wav', x_lpc, config['sample_rate'])
        utils.soundfile_writer(os.path.join(output_path, 'y_'+str(count)+'.wav'), y_lpc, config['sample_rate'])

    mse_mean = np.mean(np.asarray(snrs))
    #sdr_mean = np.mean(np.asarray(sdrs))
    print('SNR computed:',mse_mean)
    #print('SDR computed:',sdr_mean)
    # Write results to csv
    np.save(os.path.join(model_path,'mse_results_'+str(factor)), np.asarray(snrs))
    #np.save(os.path.join(model_path,'sdr_results_'+str(factor)), np.asarray(sdrs))
    results = config['results_csv']
    if not os.path.isfile(results): 
        with open(results, "w") as c: 
            csvwriter = csv.writer(c)  
            csvwriter.writerow(['model','target_bitrate','num_param','mse']) 
    
    frame = pd.read_csv(results, header=0)
    frame.set_index('model')
    d = {'model':config['model'], 'target_bitrate': config['target_bitrate'], 'num_param':str(num_param//1000)+'k', 'mse':mse_mean}
    frame = frame.append(d, ignore_index=True)
    frame = frame.loc[:, ~frame.columns.str.contains('^Unnamed')]
    frame.to_csv(results)
    
    return np.mean(np.asarray(snrs))

@ex.automain
def main(cfg):
    evaluate(config=cfg['config'])