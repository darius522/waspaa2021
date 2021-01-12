import os
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
import _pickle as pickle
from torch.utils.data import TensorDataset, DataLoader
import torchaudio

device='cuda:3'

import librosa
import librosa.display

import time
import IPython.display as ipd


import soundfile as sf
import logging

import model


m = model.Waveunet()
summary(m,(1,16384),device='cpu')