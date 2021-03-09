import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import utils
import os
import tqdm

rootPath='../../../media/sdb1/Data/ETRI_Music/'

allPaths = []
allPaths += [os.path.join(rootPath,song) for song in os.listdir(rootPath) if not os.path.isdir(os.path.join(rootPath, song))]

for path in tqdm.tqdm(allPaths):
    audio = utils.load_audio(path)
    if torch.any(torch.isnan(audio)):
        print(path)

