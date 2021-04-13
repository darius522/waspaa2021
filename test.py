import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import utils
import os
import tqdm
import csv
import numpy as np
import glob

from pydub import AudioSegment

dir_out = '../harpnet_evaluation/configs/resources/audio/mp3l_songs'

paths = glob.glob('../harpnet_evaluation/configs/resources/audio/gt_songs/*.wav')
paths = sorted(paths)

lengths = []
for path in paths:

    sound = AudioSegment.from_file(path)

    stereo_sound = AudioSegment.from_mono_audiosegments(sound, sound)
    
    stereo_sound.export(path, format='WAV')

# test_csv = './data/test_set44.csv'
# with open(test_csv, newline='') as f:
#     reader = csv.reader(f)
#     testPaths = list(reader)


# count = 0
# paths = glob.glob('../harpnet_evaluation/configs/resources/audio/gt_songs/*.wav')
# paths = sorted(paths)
# for i, path in enumerate(paths):
#     count += 1
#     audio = utils.load_audio(path, start=0, dur=None)
#     audio = torch.clone(torch.mean(audio, axis=0, keepdim=True))
#     print(audio.size(1),lengths[i])
#     new_audio = audio[:,:lengths[i]]
#     print(new_audio.size(1)==lengths[i])
#     new_audio = new_audio.permute(1,0).cpu().detach().numpy() 
#     num = os.path.splitext(path)[0].split('_')[-1].split('.')[0]

#     utils.soundfile_writer(os.path.join(dir_out, 'yn_'+str(num)+'.mp3'), new_audio, 44100)