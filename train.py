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

device='cuda:7'

import librosa
import librosa.display

import time
import IPython.display as ipd


import soundfile as sf
import logging

import model


rootPath='../../../sdc1/Data/ETRI_Music/'
allPaths=[]
allPaths+= [os.path.join(rootPath,song) for song in os.listdir(rootPath) if not os.path.isdir(os.path.join(rootPath, song))]
totLen=len(allPaths)
random.seed(0)

brokens = ["0108-00003-HMD.wav", "0063-00002-HMD.wav", "0150-00003-RMD.wav", "0122-00003-HMD.wav", "0003-00001-AMD.wav"]
ii = []
for p in range(0, totLen):
    if os.path.basename(allPaths[p]) in brokens:
        ii.append(p)
allPaths = np.delete(allPaths,ii)

random.shuffle(allPaths)
trPaths=allPaths[:np.int(totLen*.9)]
vPaths=allPaths[np.int(totLen*.9):np.int(totLen*.95)]
tePaths=allPaths[np.int(totLen*.95):]

H=60
W2=5
maxEpoch=300
BS=20
minLen=16384

m = model.Waveunet()
summary(m,(1,16384),device='cpu')

optimizer= torch.optim.Adam(m.parameters(), lr=1e-3)
maxValSNR=-1000
scale=1e-10
zeropadding=1024+7040

m.train()
random.shuffle(trPaths)
tic=time.time()

for epoch in range(maxEpoch):

    m.train()
    random.shuffle(trPaths)
    tic=time.time()
    for p in range(0, len(trPaths), BS):

        if p+BS>len(trPaths):
            BStail=len(trPaths)-p
        else:
            BStail=BS
            
        x=torch.zeros((BStail, 2, minLen), device=device)
        
        for bs in range(BStail):
            filename=trPaths[p+bs]
            waveform, sr = torchaudio.load(filename)
            waveform=torchaudio.transforms.Resample(sr, 44100)(waveform)[:,:minLen]
#             x_temp, _=librosa.load(p, sr=None, mono=False)
            x[bs,]=waveform.to(device)
        mb=torch.mean(x, axis=1, keepdim=True)
#         mbl=x[:,0,None,:]
#         mbr=x[:,1,None,:]
#         print(mb.size(), mbl.size(), mbr.size())
        y=model(torch.cat((torch.randn((BStail, 1, zeropadding), device=device)*scale, mb), 2)) #896 for W=32
#         print(yl.shape[2]-mbl.shape[2])
#         err=(-SNR(mbl.squeeze(), yl.squeeze())-SNR(mbr.squeeze(), yr.squeeze()))/2
        err=torch.mean((mb-y)**2)
        optimizer.zero_grad()
        err.backward()
        optimizer.step()                                    
        
        # if p%20 == 0 and p is not 0:
        #     with torch.no_grad():
        #         valSNR=FCN_AE_test(model, vPaths, scale, zeropadding, quantization=False, writeout=False)
        #     if valSNR > maxValSNR:
        #         maxValSNR = valSNR
        #         torch.save(model.state_dict(), 'FCN_AE_Music_LV{}.model'.format(Hc)) 
        #     toc=time.time()
        #     logging.info('ep: {0}\t err: {1:3.4}\t SNR: {2:3.4}\t max SNR: {3:3.4}\t time: {4:3.3}'.format(epoch, err.detach().cpu(), valSNR, maxValSNR, toc-tic))    
        #     tic=time.time()