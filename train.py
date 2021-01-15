import os
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
import tqdm

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
import utils

rootPath='../../../media/sdb1/Data/ETRI_Music/'
experiment_id = np.random.randint(0,1000000)

parser = argparse.ArgumentParser(description='Trainer')

parser.add_argument('--experiment-id', type=str, default=str(experiment_id))

# Dataset paramaters
parser.add_argument('--root', type=str, default=rootPath, help='root path of dataset')
parser.add_argument('--output', type=str, default="output",
                    help='provide output path base folder name')
parser.add_argument('--model', type=str, help='Path to checkpoint folder')

# Trainig Parameters
parser.add_argument('--epochs', type=int, default=300)
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
parser.add_argument('--seq-dur', type=float, default=16384, # 16384 samples
                    help='Sequence duration in seconds'
                    'value of <=0.0 will use full/variable length')
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


allPaths  = []
allPaths += [os.path.join(rootPath,song) for song in os.listdir(rootPath) if not os.path.isdir(os.path.join(rootPath, song))]
totLen    = len(allPaths)
random.seed(0)

random.shuffle(allPaths)
trPaths = allPaths[:np.int(totLen*.9)]
vPaths  = allPaths[np.int(totLen*.9):np.int(totLen*.95)]
tePaths = allPaths[np.int(totLen*.95):]

utils.dataset_items_to_csv(path=os.path.join(args.output,'testset_'+args.experiment_id+'.csv'),items=tePaths)

random.shuffle(trPaths)
tic=time.time()

def train(args, waveunet, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    waveunet.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = waveunet(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()
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
target_path = Path(args.output)
target_path.mkdir(parents=True, exist_ok=True)

train_sampler = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    **dataloader_kwargs
)

valid_sampler = torch.utils.data.DataLoader(
    valid_dataset, batch_size=1,
    **dataloader_kwargs
)

waveunet = model.Waveunet()
waveunet.to(device)
#summary(waveunet,(2,args.seq_dur),device='cpu')

optimizer = torch.optim.Adam(
    waveunet.parameters(),
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
    train_loss = train(args, waveunet, device, train_sampler, optimizer)
    valid_loss = valid(args, waveunet, device, valid_sampler)
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
            'state_dict': waveunet.state_dict(),
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
        'num_bad_epochs': es.num_bad_epochs,
        'commit': commit
    }

    with open(Path(target_path,  args.experiment_id + '.json'), 'w') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))

    train_times.append(time.time() - end)

    if stop:
        print("Apply Early Stopping")
        break


if __name__ == "__main__":
    main()