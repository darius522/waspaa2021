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

from utils import (normalize_audio, model_dic)

import torchaudio

parser = argparse.ArgumentParser(description='Trainer')

parser.add_argument('--main-dir', type=str, default='output')
parser.add_argument('--model-name', type=str, default='waveunet_skip')
parser.add_argument('--model-id', type=str, default='909074')
# Model Parameters
parser.add_argument('--seq-dur', type=float, default=16384)
parser.add_argument('--nb-channels', type=int, default=1)
parser.add_argument('--sample-rate', type=int, default=44100)
parser.add_argument('--overlap', type=int, default=64)
parser.add_argument('--window', type=str, default='hann')
parser.add_argument('--device', type=str, default='cpu')

args, _ = parser.parse_known_args()

def compute_snr(x, y):
    eps = 1e-20
    ml  = np.minimum(len(x), len(y))
    return 10 * torch.log10((torch.sum(x[:ml]**2)  / torch.sum((x[:ml]-y[:ml])**2) + eps) + eps)   

def load_model(
    model_name=args.model_name, 
    model_id=args.model_id, 
    device=args.device,
    model_dic=model_dic
):

    model_path = os.path.join(args.main_dir,model_name+'/'+model_id)
    print(model_path)
    # load model from disk
    with open(Path(model_path, model_id + '.json'), 'r') as stream:
        results = json.load(stream)

    model_path = next(Path(model_path).glob("%s*.pth" % model_id))
    print(model_path)
    state = torch.load(
        model_path,
        map_location=device
    )

    model = models.Waveunet(
        n_ch=results['args']['nb_channels'],
        model=model_dic[results['args']['model']]

    )

    model.quant_active = True
    for m in model.skip_encoders:
        m.quant_active = True

    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return model

def prepare_audio(audio):

    audio_len = audio.size()[-1]
    hop_size  = args.seq_dur - args.overlap
    num_frames = math.ceil(audio_len / hop_size)

    prep_audio = torch.zeros(num_frames, args.nb_channels, args.seq_dur)
    timestamps = torch.zeros(num_frames,2)

    end = 0
    for i in range(num_frames):

        start = (i * hop_size)
        end   = start + args.seq_dur

        timestamps[i,0] = start
        timestamps[i,1] = end

        if audio_len > end:
            prep_audio[i,:,:] = torch.clone(audio[:,start:end])
        else:
            last = args.seq_dur - (end - audio_len)
            prep_audio[i,:,:last] = torch.clone(audio[:,start:start+last])
            prep_audio[i,:,last:] = 0
    
    return prep_audio, timestamps

def overlap_add(audio, timestamps, device):

    audio_ = torch.clone(audio)

    num_frames = audio_.size()[0]
    target_len = num_frames * (args.seq_dur - args.overlap) + args.overlap
    y = torch.zeros(args.nb_channels, target_len)

    hann = torch.hann_window(args.overlap*2, periodic=True)

    for i in range(num_frames):

        start = int(timestamps[i,0].item())
        end   = int(timestamps[i,1].item())

        chunk = torch.clone(audio_[i,:,:])
        for j in range(args.nb_channels):
            chunk[j,:args.overlap]  = chunk[j,:args.overlap] * hann[:args.overlap]
            chunk[j,-args.overlap:] = chunk[j,-args.overlap:] * hann[args.overlap:]

        y[:,start:end] = y[:,start:end] + chunk

    return y
        

def inference(
    audio,
    model,
    device=args.device
):

    x, timestamps = prepare_audio(audio)
    x_new = overlap_add(x,timestamps,device=device)

    y_tmp = model(x)
    y_tmp_2 = y_tmp.detach().clone()

    y = overlap_add(y_tmp_2,timestamps,device=device)

    return x_new, y

def evaluate_model(model_name='',model_id=''):

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    print("Using GPU:", use_cuda)

    with open('./data/test_set.csv', newline='') as f:
        reader = csv.reader(f)
        testPaths = list(reader)

    testPaths = [path for sublist in testPaths for path in sublist]
    
    model = load_model(
    model_name=model_name,
    model_id=model_id,
    device=device
    )

    errors = []
    count = 0
    for test_path in tqdm.tqdm(testPaths[:2]):

        count += 1
        audio = utils.load_audio(test_path, start=0, dur=None, sr=args.sample_rate)
        if args.nb_channels == 1:
            audio_mono = torch.clone(torch.mean(audio, axis=0, keepdim=True))
    
        x, y = inference(model=model,audio=audio_mono,device=device)
        errors.append(compute_snr(x, y))
        utils.soundfile_writer(os.path.join(args.main_dir,model_name+'/'+model_id+'/'+'x'+str(count)+'.wav'), x.cpu().permute(1,0).detach().numpy(), 44100)
        utils.soundfile_writer(os.path.join(args.main_dir,model_name+'/'+model_id+'/'+'y'+str(count)+'.wav'), y.cpu().permute(1,0).detach().numpy(), 44100)

    print(model_name,'/',model_id,'evaluated with snr mean:',np.mean(np.asarray(errors)))

evaluate_model(model_name='waveunet_no_skip',model_id='131595')