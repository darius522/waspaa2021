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

from models import Model
from utils import normalize_audio

model_dic = {
    'waveunet_no_skip': Model.waveunet_no_skip,
    'waveunet_skip': Model.waveunet_skip,
    'waveunet_enc_skip': Model.waveunet_enc_skip
}

parser = argparse.ArgumentParser(description='Trainer')

parser.add_argument('--main-dir', type=str, default='output')
parser.add_argument('--model-name', type=str, default='waveunet_skip')
parser.add_argument('--model-id', type=str, default='909074')
# Model Parameters
parser.add_argument('--seq-dur', type=float, default=16384)
parser.add_argument('--nb-channels', type=int, default=1)
parser.add_argument('--overlap', type=int, default=64)
parser.add_argument('--window', type=str, default='hann')
parser.add_argument('--device', type=str, default='cpu')

args, _ = parser.parse_known_args()

def load_model(
    model_name=args.model_name, 
    model_id=args.model_id, 
    device=args.device,
    model_dic=model_dic
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
        n_ch=results['args']['nb_channels'],
        model=model_dic[results['args']['model']]
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
    timestamps = torch.empty(num_frames,2)

    end = 0
    for i in range(num_frames):

        start = (i * hop_size)
        end   = start + args.seq_dur

        timestamps[i,0] = start
        timestamps[i,1] = end

        if audio_len > end:
            prep_audio[i,:,:] = torch.clone(audio[:,start:end])#torch.narrow(audio,1,start,args.seq_dur)
        else:
            last = args.seq_dur - (end - audio_len)
            prep_audio[i,:,:last] = torch.clone(audio[:,start:start+last])#torch.narrow(audio,1,start,last)
            prep_audio[i,:,last:] = 0
    
    return prep_audio, timestamps

def overlap_add(audio, timestamps, device):
    num_frames = audio.size()[0]
    target_len = num_frames * (args.seq_dur - args.overlap) + args.overlap
    y = torch.empty(args.nb_channels, target_len)

    hann = torch.hann_window(args.overlap*2, periodic=True)

    for i in range(num_frames):

        start = int(timestamps[i,0].item())
        end   = int(timestamps[i,1].item())

        chunk = torch.clone(audio[i,:,:])
        for j in range(args.nb_channels):
            chunk[j,:args.overlap]  *= hann[:args.overlap]
            chunk[j,-args.overlap:] *= hann[args.overlap:]
            
        y[:,start:end] += chunk

    return y
        

def inference(
    audio,
    model_name=args.model_name, 
    model_id=args.model_id, 
    device=args.device
):

    model = load_model(
        model_name=model_name,
        model_id=model_id,
        device=device
        )

    x, timestamps = prepare_audio(audio)
    x_new = overlap_add(x,timestamps,device=device)

    y_tmp = model(x)

    y = overlap_add(y_tmp,timestamps,device=device)

    # from matplotlib import pyplot as plt
    # plt.figure()
    # _ = plt.plot(x_new)
    # plt.savefig('./x.png')
    # plt.clf()
    # plt.figure()
    # _ = plt.plot(y)
    # plt.savefig('./y.png')
    utils.soundfile_writer('./y'+str(args.overlap)+'.wav', y.cpu().permute(1,0).detach().numpy(), 44100)
    return x_new, y

use_cuda = torch.cuda.is_available()
device = torch.device(args.device if use_cuda else "cpu")
print("Using GPU:", use_cuda)

with open(os.path.join(args.main_dir,args.model_name+'/'+args.model_id+'/'+'test_set.csv'), newline='') as f:
    reader = csv.reader(f)
    testPaths = list(reader)

testPaths = [path for sublist in testPaths for path in sublist]

audio = utils.torchaudio_loader(testPaths[1])
if args.nb_channels == 1:
    audio = torch.mean(audio, axis=0, keepdim=True)

x, y = inference(audio=audio,device=device)

#y = normalize_audio(torch.min(x),torch.max(x),y)
print("min: "+ str(torch.min(x)))
print("max: "+ str(torch.max(x)))
print("min: "+ str(torch.min(y)))
print("max: "+ str(torch.max(y)))
utils.soundfile_writer('./y_broken'+str(args.overlap)+'.wav', y.cpu().permute(1,0).detach().numpy(), 44100)