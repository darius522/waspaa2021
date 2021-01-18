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

parser = argparse.ArgumentParser(description='Trainer')

parser.add_argument('--main-dir', type=str, default='output')
parser.add_argument('--model-name', type=str, default='waveunet')
parser.add_argument('--model-id', type=str, default='517737')
# Model Parameters
parser.add_argument('--seq-dur', type=float, default=16384)
parser.add_argument('--nb-channels', type=int, default=1)
parser.add_argument('--overlap', type=int, default=512)
parser.add_argument('--window', type=str, default='hann')

args, _ = parser.parse_known_args()

def load_model(
    model_name=args.model_name, 
    model_id=args.model_id, 
    device='cuda:7'
):

    model_path = os.path.join(args.main_dir,model_name+'/'+model_id)
    # load model from disk
    with open(Path(model_path, model_id + '.json'), 'r') as stream:
        results = json.load(stream)

    model_path = next(Path(model_path).glob("%s*.pth" % model_id))
    state = torch.load(
        model_path,
        map_location=device
    )

    model = models.Waveunet(
        n_ch=results['args']['nb_channels']
    )

    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return model

def prepare_audio(audio):

    audio_len = audio.size()[-1]
    hop_size  = args.seq_dur - args.overlap
    num_frames = math.ceil(audio_len / hop_size)

    prep_audio = torch.empty(num_frames, args.nb_channels, args.seq_dur)

    for i in range(num_frames):

        start = (i * hop_size)
        end   = start + args.seq_dur

        if audio_len > end:
            slic = torch.narrow(audio,1,start,args.seq_dur)
            prep_audio[i,:,:] = torch.narrow(audio,1,start,args.seq_dur)
        else:
            last = args.seq_dur - (end - audio_len)
            prep_audio[i,:,:last] = torch.narrow(audio,1,start,last)
            prep_audio[i,:,last:] = 0
    
    return prep_audio

def overlap_add(audio, device):
    num_frames = audio.size()[0]
    target_len = num_frames * (args.seq_dur - args.overlap) + args.overlap
    y = torch.empty(args.nb_channels, target_len).to(device)

    hann = torch.hann_window(args.overlap, periodic=True).to(device)
    half_hann = int(args.overlap/2)

    for i in range(num_frames):

        start = i * (args.seq_dur - args.overlap)
        end   = start + args.seq_dur

        for j in range(args.nb_channels):
            audio[i,j,:half_hann] *= hann[:half_hann]
            audio[i,j,-half_hann:] *= hann[half_hann:]
            y[j,start:end] += audio[i,j,:]

    return y
        

def inference(
    audio,
    model_name=args.model_name, 
    model_id=args.model_id, 
    device='cuda:7'
):
    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)

    model = load_model(
        model_name=model_name,
        model_id=model_id,
        device=device
        )

    x = prepare_audio(audio)
    x = x.to(device)
    y_tmp = model(x).to(device)

    y = overlap_add(y_tmp,device=device)

    return y

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:7" if use_cuda else "cpu")
print("Using GPU:", use_cuda)

with open(os.path.join(args.main_dir,args.model_name+'/'+args.model_id+'/'+'test_set.csv'), newline='') as f:
    reader = csv.reader(f)
    testPaths = list(reader)

testPaths = [path for sublist in testPaths for path in sublist]

audio = utils.torchaudio_loader(testPaths[0])
if args.nb_channels == 1:
    audio = torch.mean(audio, axis=0, keepdim=True)

y = inference(audio=audio,device=device)

utils.soundfile_writer('./test_x.wav', audio.cpu().permute(1,0).detach().numpy(), 44100)
utils.soundfile_writer('./test_y.wav', y.cpu().permute(1,0).detach().numpy(), 44100)