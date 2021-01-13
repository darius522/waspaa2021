import os
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path

import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
import _pickle as pickle
from torch.utils.data import TensorDataset, DataLoader
import torchaudio

import librosa
import librosa.display

import time
import IPython.display as ipd


import soundfile as sf
import logging

import argparse
import model
import data

parser = argparse.ArgumentParser(description='Open Unmix Trainer')

# which target do we want to train?
parser.add_argument('--target', type=str, default='vocals',
                    help='target source (will be passed to the dataset)')

# Dataset paramaters
parser.add_argument('--dataset', type=str, default="musdb",
                    choices=[
                        'musdb', 'aligned', 'sourcefolder',
                        'trackfolder_var', 'trackfolder_fix'
                    ],
                    help='Name of the dataset.')
parser.add_argument('--root', type=str, help='root path of dataset')
parser.add_argument('--output', type=str, default="open-unmix",
                    help='provide output path base folder name')
parser.add_argument('--model', type=str, help='Path to checkpoint folder')

# Trainig Parameters
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, defaults to 1e-3')
parser.add_argument('--patience', type=int, default=140,
                    help='maximum number of epochs to train (default: 140)')
parser.add_argument('--lr-decay-patience', type=int, default=80,
                    help='lr decay patience for plateau scheduler')
parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                    help='gamma of learning rate scheduler decay')
parser.add_argument('--weight-decay', type=float, default=0.00001,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

# Model Parameters
parser.add_argument('--seq-dur', type=float, default=6.0,
                    help='Sequence duration in seconds'
                    'value of <=0.0 will use full/variable length')
parser.add_argument('--unidirectional', action='store_true', default=False,
                    help='Use unidirectional LSTM instead of bidirectional')
parser.add_argument('--nfft', type=int, default=4096,
                    help='STFT fft size and window size')
parser.add_argument('--nhop', type=int, default=1024,
                    help='STFT hop size')
parser.add_argument('--hidden-size', type=int, default=512,
                    help='hidden size parameter of dense bottleneck layers')
parser.add_argument('--bandwidth', type=int, default=16000,
                    help='maximum model bandwidth in herz')
parser.add_argument('--nb-channels', type=int, default=2,
                    help='set number of channels for model (1, 2)')
parser.add_argument('--nb-workers', type=int, default=0,
                    help='Number of workers for dataloader.')

# Misc Parameters
parser.add_argument('--quiet', action='store_true', default=False,
                    help='less verbose during training')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args, _ = parser.parse_known_args()
print(args.nb_channels)

device = torch.device("cuda")


rootPath='../../../media/sdb1/Data/ETRI_Music/'
allPaths=[]
allPaths+= [os.path.join(rootPath,song) for song in os.listdir(rootPath) if not os.path.isdir(os.path.join(rootPath, song))]
totLen=len(allPaths)
random.seed(0)

random.shuffle(allPaths)
trPaths=allPaths[:np.int(totLen*.9)]
vPaths=allPaths[np.int(totLen*.9):np.int(totLen*.95)]
tePaths=allPaths[np.int(totLen*.95):]

H=60
W2=5
maxEpoch=300
BS=20
minLen=16384

maxValSNR=-1000
scale=1e-10
zeropadding=1024+7040

random.shuffle(trPaths)
tic=time.time()

def train(args, m, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        Y_hat = unmix(x)
        Y = unmix.transform(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
    return losses.avg

use_cuda = torch.cuda.is_available()
print("Using GPU:", use_cuda)
dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}

repo_dir = os.path.abspath(os.path.dirname(__file__))

# use jpg or npy
torch.manual_seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

train_dataset, valid_dataset, args = data.load_datasets(parser, args)

# create output dir if not exist
target_path = Path("./output")
target_path.mkdir(parents=True, exist_ok=True)

# train_sampler = torch.utils.data.DataLoader(
#     train_dataset, batch_size=BS, shuffle=True,
#     **dataloader_kwargs
# )
# valid_sampler = torch.utils.data.DataLoader(
#     valid_dataset, batch_size=1,
#     **dataloader_kwargs
# )

# waveunet = model.Waveunet().to(device)
# #summary(m,(1,16384),device='cpu')

# optimizer = torch.optim.Adam(
#     unmix.parameters(),
#     lr=args.lr,
#     weight_decay=args.weight_decay
# )

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     factor=args.lr_decay_gamma,
#     patience=args.lr_decay_patience,
#     cooldown=10
# )

# es = utils.EarlyStopping(patience=args.patience)

# t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
# train_losses = []
# valid_losses = []
# train_times = []
# best_epoch = 0

# for epoch in t:
#     t.set_description("Training Epoch")
#     end = time.time()
#     train_loss = train(args, waveunet, device, train_sampler, optimizer)
#     valid_loss = valid(args, waveunet, device, valid_sampler)
#     scheduler.step(valid_loss)
#     train_losses.append(train_loss)
#     valid_losses.append(valid_loss)

#     t.set_postfix(
#         train_loss=train_loss, val_loss=valid_loss
#     )

#     stop = es.step(valid_loss)

#     if valid_loss == es.best:
#         best_epoch = epoch

#     utils.save_checkpoint({
#             'epoch': epoch + 1,
#             'state_dict': waveunet.state_dict(),
#             'best_loss': es.best,
#             'optimizer': optimizer.state_dict(),
#             'scheduler': scheduler.state_dict()
#         },
#         is_best=valid_loss == es.best,
#         path=target_path,
#         target=args.target
#     )

#     # save params
#     params = {
#         'epochs_trained': epoch,
#         'args': vars(args),
#         'best_loss': es.best,
#         'best_epoch': best_epoch,
#         'train_loss_history': train_losses,
#         'valid_loss_history': valid_losses,
#         'train_time_history': train_times,
#         'num_bad_epochs': es.num_bad_epochs,
#         'commit': commit
#     }

#     with open(Path(target_path,  args.target + '.json'), 'w') as outfile:
#         outfile.write(json.dumps(params, indent=4, sort_keys=True))

#     train_times.append(time.time() - end)

#     if stop:
#         print("Apply Early Stopping")
#         break


# if __name__ == "__main__":
# main()