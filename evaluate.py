import os
import torch
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

parser = argparse.ArgumentParser(description='Trainer')

parser.add_argument('--main-dir', type=str, default='output')
parser.add_argument('--model-name', type=str, default='waveunet')
parser.add_argument('--model-id', type=str, default='517737')
# Model Parameters
parser.add_argument('--seq-dur', type=float, default=16384)
parser.add_argument('--nb-channels', type=int, default=1)
parser.add_argument('--overlap', type=int, default=32)
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

    # 1. calculate the number of frame needed
    audio_len = audio.size()[-1]
    num_frames = math.ceil(audio_len / (args.seq_dur - args.overlap))

    prep_audio = torch.empty(num_frames, args.nb_channels, args.seq_dur)

    for i in range(num_frames):
        

def overlap_add(audio):


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
    y_tmp = model(x).to(device)

    y = overlap_add(y_tmp)

    return y


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:7" if use_cuda else "cpu")
print("Using GPU:", use_cuda)