import os
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
import tqdm
import json

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
import models
import data
import utils

rootPath='../../../media/sdb1/Data/ETRI_Music/'
experiment_id = np.random.randint(0,1000000)

parser = argparse.ArgumentParser(description='Trainer')

parser.add_argument('--experiment-id', type=str, default=str(experiment_id))
parser.add_argument('--model', type=str, default="waveunet")

# Dataset paramaters
parser.add_argument('--root', type=str, default=rootPath, help='root path of dataset')
parser.add_argument('--output', type=str, default="output",
                    help='provide output path base folder name')

# Trainig Parameters
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--num-its', type=int, default=64)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01,
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
parser.add_argument('--seq-dur', type=float, default=16384,
                    help='Sequence duration in seconds'
                    'value of <=0.0 will use full/variable length')
parser.add_argument('--zero-pad', type=float, default=1024+7040, 
                    help='Optional zero-padding to be added to batch')
parser.add_argument('--nb-channels', type=int, default=1,
                    help='set number of channels for model (1, 2)')
parser.add_argument('--nb-workers', type=int, default=0,
                    help='Number of workers for dataloader.')

# Misc Parameters
parser.add_argument('--quiet', action='store_true', default=False,
                    help='less verbose during training')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args, _ = parser.parse_known_args()


allPaths  = []
allPaths += [os.path.join(rootPath,song) for song in os.listdir(rootPath) if not os.path.isdir(os.path.join(rootPath, song))]
totLen    = len(allPaths)
random.seed(0)

random.shuffle(allPaths)
trPaths = allPaths[:np.int(totLen*.9)]
vPaths  = allPaths[np.int(totLen*.9):np.int(totLen*.95)]
tePaths = allPaths[np.int(totLen*.95):]

random.shuffle(trPaths)
tic=time.time()

def train(args, model, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    model.train()
    i_bar = tqdm.tqdm(range(args.num_its), disable=args.quiet)
    b_bar = tqdm.tqdm(train_sampler, disable=args.quiet)

    for _ in i_bar:
        i_bar.set_description("Training Iteration")
        for x, _ in b_bar:
            b_bar.set_description("Training Batch")
            x = x.to(device)
            optimizer.zero_grad()
            if args.model == 'unet':
                y_hat = model(torch.cat((torch.randn((args.batch_size, args.nb_channels, args.zero_pad), device=device)*1e-10, x), 2))
            else:
                y_hat = model(x)
            loss = torch.nn.functional.mse_loss(y_hat, x)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), x.size(1))

    return losses.avg

def valid(args, model, device, valid_sampler):
    losses = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            y_hat = model(x).to(device)
            loss = torch.nn.functional.mse_loss(y_hat, y)
            losses.update(loss.item(), y.size(1))
        return losses.avg

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:7" if use_cuda else "cpu")
print("Using GPU:", use_cuda)
dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}

repo_dir = os.path.abspath(os.path.dirname(__file__))

# use jpg or npy
torch.manual_seed(args.seed)
random.seed(args.seed)

train_dataset, valid_dataset, args = data.load_datasets(parser, args, train=trPaths, valid=vPaths)

# create output dir if not exist
target_path = Path(os.path.join(args.output,args.model+"/"+args.experiment_id))
target_path.mkdir(parents=True, exist_ok=True)

utils.dataset_items_to_csv(path=os.path.join(args.output,args.model+'/'+'test_'+args.experiment_id+'.csv'),items=tePaths)

train_sampler = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
    **dataloader_kwargs
)

valid_sampler = torch.utils.data.DataLoader(
    valid_dataset, batch_size=1, drop_last=True,
    **dataloader_kwargs
)

if args.model == 'waveunet':
    model = models.Waveunet(n_ch=args.nb_channels).to(device)
elif args.model == 'unet':
    args.seq_dur = 350912
    model = models.U_Net(H=60, Hc=4, Hskip=4, W1=32, W2=5).to(device)

#summary(model,(args.nb_channels,args.seq_dur),device='cpu')

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=args.lr_decay_gamma,
    patience=args.lr_decay_patience,
    cooldown=10
)

es = utils.EarlyStopping(patience=args.patience)

t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
train_losses = []
valid_losses = []
train_times = []
best_epoch = 0

for epoch in t:
    t.set_description("Training Epoch")
    end = time.time()
    train_loss = train(args, model, device, train_sampler, optimizer)
    valid_loss = valid(args, model, device, valid_sampler)
    scheduler.step(valid_loss)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    t.set_postfix(
        train_loss=train_loss, val_loss=valid_loss
    )

    stop = es.step(valid_loss)

    if valid_loss == es.best:
        best_epoch = epoch

    utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': es.best,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        },
        is_best=valid_loss == es.best,
        path=target_path,
        target=args.experiment_id
    )

    # save params
    params = {
        'epochs_trained': epoch,
        'args': vars(args),
        'best_loss': es.best,
        'best_epoch': best_epoch,
        'train_loss_history': train_losses,
        'valid_loss_history': valid_losses,
        'train_time_history': train_times,
        'num_bad_epochs': es.num_bad_epochs
    }

    with open(Path(target_path,  args.experiment_id + '.json'), 'w') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))

    train_times.append(time.time() - end)

    if stop:
        print("Apply Early Stopping")
        break


if __name__ == "__main__":
    main()