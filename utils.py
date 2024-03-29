import shutil
import torch
import os
import numpy as np
import csv
import json
from matplotlib import pyplot as plt
import torch.nn as nn

import torchaudio
import soundfile
from audiolazy import *

from scipy.interpolate import interp1d
from enum import Enum

def normalize_audio(min, max, audio):
    return (max - min) * ((audio - torch.min(audio)) / (torch.max(audio) - torch.min(audio))) + min

def get_uniform_distribution(num_bins):
    t = torch.empty(num_bins)
    return nn.init.uniform_(t, a=-0.8, b=0.8)

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    
    return smoothed

def entropy_to_bitrate(entropy, sr, window_size, overlap, bottleneck_dim):
    return (sr / (window_size-overlap)) * entropy * np.prod(bottleneck_dim)

def bitrate_to_entropy(bitrate, sr, window_size, overlap, bottleneck_dim):
    return bitrate * (window_size - overlap) / (sr * np.prod(bottleneck_dim))

def plot_loss_to_png(json_file):
    with open(json_file) as jsonF: 
        j = json.load(jsonF) 
        train_hist = list(j["train_loss_history"])
        valid_hist = list(j["valid_loss_history"])

        train_hist = smooth(train_hist,0.9)
        plt.plot(train_hist,color='blue')
        plt.savefig(os.path.splitext(json_file)[0]+'_train_loss'+'.png')

        plt.clf()
        plt.cla()

        valid_hist = smooth(valid_hist,0.9)
        plt.plot(valid_hist,color='orange')
        plt.savefig(os.path.splitext(json_file)[0]+'_valid_loss'+'.png')

def dataset_items_to_csv(path, items):
    with open(path, 'w') as f: 
        write = csv.writer(f) 
        for i in items:
            write.writerow([i]) 

def _sndfile_available():
    try:
        import soundfile
    except ImportError:
        return False

    return True


def _torchaudio_available():
    try:
        import torchaudio
    except ImportError:
        return False

    return True


def get_loading_backend():
    if _torchaudio_available():
        return torchaudio_loader

    if _sndfile_available():
        return soundfile_loader


def get_info_backend():
    if _torchaudio_available():
        return torchaudio_info

    if _sndfile_available():
        return soundfile_info

def soundfile_writer(path, audio, sr):
    soundfile.write(path, audio, sr)

def soundfile_info(path):
    info = {}
    sfi = soundfile.info(path)
    info['samplerate'] = sfi.samplerate
    info['samples'] = int(sfi.duration * sfi.samplerate)
    info['duration'] = sfi.duration
    return info


def soundfile_loader(path, start=0, dur=None):
    # get metadata
    # check if dur is none
    if dur:
        # stop in soundfile is calc in samples, not seconds
        stop = start + dur
    else:
        # set to None for reading complete file
        stop = dur

    audio, _ = soundfile.read(
        path,
        always_2d=True,
        start=start,
        stop=stop
    )
    return torch.FloatTensor(audio.T)


def torchaudio_info(path):
    # get length of file in samples
    info = {}
    si, _ = torchaudio.info(str(path))
    info['samplerate'] = si.rate
    info['samples'] = si.length // si.channels
    info['duration'] = info['samples'] / si.rate
    return info


def torchaudio_loader(path, start=0, dur=None):
    info = torchaudio_info(path)
    # loads the full track duration
    if dur is None:
        sig, rate = torchaudio.load(path)
        return sig, rate
        # otherwise loads a random excerpt
    else:
        start = int(start)
        dur   = int(dur)
        sig, rate = torchaudio.load(
            path, num_frames=dur, offset=start
        )
        return sig, rate


def load_info(path):
    loader = get_info_backend()
    return loader(path)


def load_audio(path, start=0, dur=None, sr=44100):
    loader = get_loading_backend()
    audio, fs = loader(path, start=start, dur=dur)
    if not fs == sr:
        audio = torchaudio.transforms.Resample(fs, sr)(audio)

    return audio


def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(
        0, float(rate) / 2, n_fft // 2 + 1,
        endpoint=True
    )

    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def save_checkpoint(
    state, is_best, path, target
):
    # save full checkpoint including optimizer
    torch.save(
        state,
        os.path.join(path, target + '.chkpnt')
    )
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, target + '.pth')
        )

def vec2mat(x, framesize, overlap, window=[]): 
    assert len(x.shape)==1
    hopsize=framesize-overlap
    nFr=x.shape[0]//hopsize
    if x.shape[0] >= nFr*hopsize:
        x=np.concatenate((x, np.zeros(nFr*hopsize+framesize-x.shape[0])))
        nFr=nFr+1
    X=np.zeros((framesize, nFr))    
    for i in range(nFr):
        X[:,i]=x[i*hopsize:i*hopsize+framesize].copy()
    if len(window) > 0:
        X[:overlap,:]*=window[:overlap]
        X[-overlap:,:]*=window[-overlap:]
    return X

def mat2vec(X, framesize, overlap, window=[]): 
    hopsize=framesize-overlap
    nFr=X.shape[1]
    x=np.zeros((nFr-1)*hopsize+framesize)
    if len(window) > 0:
        X[:overlap,:]*=window[:overlap]
        X[-overlap:,:]*=window[-overlap:]
    for i in range(nFr):
        x[i*hopsize:i*hopsize+framesize]+=X[:,i].copy()
    return x

def LPC_synthesis(LPCcoef, residual):
    analysis_filt = ZFilter(list(LPCcoef), [1])
#     print(analysis_filt)
    synth_filt = 1 / analysis_filt
#     print(synth_filt)
    unstable=True
    for i in range(10000): # check thte synth filter stability for 10000 times
        if not lsf_stable(synth_filt):
            LPCcoef*=.95
            analysis_filt = ZFilter(list(LPCcoef), [1])
            synth_filt = 1 / analysis_filt
        else:
            unstable=False
            break
        
    yh=synth_filt(residual)
    yh=np.array(list(yh))
    return yh, unstable

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta