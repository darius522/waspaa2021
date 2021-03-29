import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import utils
import os
import tqdm

audio = utils.load_audio('../../../media/sdb1/Data/0809-00001-LFD_LPC_residual.wav')
audio*=10
print(audio)
utils.soundfile_writer('test.wav',audio.permute(1,0),32000)

