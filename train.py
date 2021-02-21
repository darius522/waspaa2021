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
from tensorboardX import SummaryWriter
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

# Local modules
import models
import data
import utils
import evaluate

rootPath='../../../media/sdb1/Data/ETRI_Music/'
experiment_id = np.random.randint(0,1000000)

parser = argparse.ArgumentParser(description='Trainer')

parser.add_argument('--experiment-id', type=str, default=str(experiment_id))
parser.add_argument('--model', type=str, default="waveunet_no_skip")
parser.add_argument('--load-ckpt', type=str, default='558920')
parser.add_argument('--message', type=str, default='layer=7, w=12, num_bins=32, entropy_target=1.5, summary: waveunet no skip')

# Dataset paramaters
parser.add_argument('--root', type=str, default=rootPath, help='root path of dataset')
parser.add_argument('--output', type=str, default="output",
                    help='provide output path base folder name')

# Trainig Parameters
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--num-its', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=16)

# Hyper-parameters
# Quant/Entropy
parser.add_argument('--quant', type=bool, default=False)
parser.add_argument('--quant-active', type=int, default=1)
parser.add_argument('--target-bitrate', type=int, default=64000,
                    help='target bitrate. by default the bitrate of 44.1 mono audio = 705,600')
parser.add_argument('--bitrate-fuzz', type=int, default=450,
                    help='amount of bitrate fuzz allowed around the target bitrate before adjusting tau')

parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate, defaults to 1e-4')
parser.add_argument('--patience', type=int, default=300,
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
parser.add_argument('--sample-rate', type=int, default=44100)
parser.add_argument('--nb-workers', type=int, default=0,
                    help='Number of workers for dataloader.')

# Misc Parameters
parser.add_argument('--quiet', action='store_true', default=False,
                    help='less verbose during training')
parser.add_argument('--device', action='store_true', default='cuda:6',
                    help='cpu or cuda')

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

def train(args, model, device, train_sampler, optimizer, writer, epoch):
    total_losses      = utils.AverageMeter()
    perceptual_losses = utils.AverageMeter()
    entropy_losses    = utils.AverageMeter()
    entropy_loss      = torch.Tensor([0]).to(device)
    model.train()

    it_bar = tqdm.tqdm(range(args.num_its), disable=args.quiet, desc='Iteration',position=0)
    batch_bar = tqdm.tqdm(train_sampler, disable=args.quiet, desc='Batch',position=1)
    torch.autograd.set_detect_anomaly(True)
    # Total num it: num_its x batch_size x training examples
    for it in it_bar:
        #it_bar.set_description("Training Iteration")
        for i, x in enumerate(batch_bar):

            x = x.to(device)
            optimizer.zero_grad()
            y_hat = model(x)

            perceptual_loss = F.mse_loss(y_hat, x)
            if model.quant_active == True:
                entropy_loss = model.entropy_loss()

            total_loss = (perceptual_loss + entropy_loss)
            total_loss.backward(retain_graph=True)
            optimizer.step()

            perceptual_losses.update(perceptual_loss.item())
            entropy_losses.update(entropy_loss.item())
            total_losses.update(total_loss.item(), x.size(1))

        if model.quant_active == True:
            print("entropy loss:",entropy_loss)
            print("perceptual loss:",perceptual_loss)
            print("entropy avg:",model.get_overall_entropy_avg())

    return perceptual_losses.avg, entropy_losses.avg, total_losses.avg

def valid(args, model, device, valid_sampler, writer, epoch):
    total_losses      = utils.AverageMeter()
    perceptual_losses = utils.AverageMeter()
    entropy_losses    = utils.AverageMeter()

    model.eval()
    with torch.no_grad():

        for x in valid_sampler:

            x = x.to(device)
            y_hat = model(x)

            perceptual_loss = F.mse_loss(y_hat, x)
            entropy_loss    = model.entropy_loss()
            total_loss      = (perceptual_loss + entropy_loss)

            perceptual_losses.update(perceptual_loss.item())
            entropy_losses.update(entropy_loss.item())
            total_losses.update(total_loss.item(), x.size(1))

        model.entropy_control_update()

    return perceptual_losses.avg, entropy_losses.avg, total_losses.avg

use_cuda = torch.cuda.is_available()
device = torch.device(args.device if use_cuda else "cpu")
print("Using GPU:", use_cuda)
dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}

torch.manual_seed(args.seed)
random.seed(args.seed)

train_dataset, valid_dataset, args = data.load_datasets(parser, args, train=trPaths, valid=vPaths)

# create output dir / log dir if not exist
target_path = Path(os.path.join(args.output,args.model+"/"+args.experiment_id))
target_path.mkdir(parents=True, exist_ok=True)
log_dir = Path(os.path.join(target_path,'log_dir'))
log_dir.mkdir(parents=True, exist_ok=True)

utils.dataset_items_to_csv(path=os.path.join(args.output,args.model+"/"+args.experiment_id+"/"+"test_set.csv"),items=tePaths)
utils.dataset_items_to_csv(path=os.path.join(args.output,args.model+"/"+args.experiment_id+"/"+"train_set.csv"),items=trPaths)
utils.dataset_items_to_csv(path=os.path.join(args.output,args.model+"/"+args.experiment_id+"/"+"val_set.csv"),items=vPaths)

train_sampler = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
    **dataloader_kwargs
)

valid_sampler = torch.utils.data.DataLoader(
    valid_dataset, batch_size=1, drop_last=True,
    **dataloader_kwargs
)


model = models.Waveunet(
    n_ch=args.nb_channels,
    model=utils.model_dic[args.model],
    target_entropy=utils.bitrate_to_entropy(args.target_bitrate,args.sample_rate),
    entropy_fuzz=utils.bitrate_to_entropy(args.bitrate_fuzz,args.sample_rate))

writer = SummaryWriter(log_dir)
#summary(model,(args.nb_channels,args.seq_dur),device='cpu')
print("Target bitrate set to:")
print("Model loaded with num. param:", sum(p.numel() for p in model.parameters() if p.requires_grad))
for name, param in model.named_parameters():
    if param.requires_grad and 'quant' in name:
        print(name, param.data)

model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr
)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     factor=args.lr_decay_gamma,
#     patience=args.lr_decay_patience,
#     cooldown=10
# )

es = utils.EarlyStopping(patience=args.patience)

t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
train_losses = []
valid_losses = []
train_times = []
best_epoch = 0

for epoch in t:

    t.set_description("Training Epoch")
    end = time.time()
    print("epoch:",epoch,"\nexperiment:",args.experiment_id,"\nmessage:",args.message)
    if epoch == args.quant_active:
        model.quant_active = True
        for m in model.skip_encoders:
            m.quant_active = True

    perc_train_loss, ent_train_loss, train_loss = train(args, model, device, train_sampler, optimizer, writer, epoch)
    perc_val_loss, ent_val_loss, valid_loss     = valid(args, model, device, valid_sampler, writer, epoch)
    #scheduler.step(valid_loss)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar("perceptual_train_loss", perc_train_loss, epoch)
    writer.add_scalar("perceptual_valid_loss", perc_val_loss, epoch)
    writer.add_scalar("entropy_train_loss", ent_train_loss, epoch)
    writer.add_scalar("entropy_valid_loss", ent_val_loss, epoch)
    writer.add_scalar("total_train_loss", train_loss, epoch)
    writer.add_scalar("total_valid_loss", valid_loss, epoch)
    writer.add_histogram('quantization bins', model.state_dict()['quant_bins'].clone().cpu().data.numpy(), epoch)
    writer.add_scalar('quantization alpha', model.state_dict()['quant_alpha'].clone().cpu().data, epoch)

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
            'optimizer': optimizer.state_dict()
            #'scheduler': scheduler.state_dict()
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

    # Post epoch business
    with open(Path(target_path,  args.experiment_id + '.json'), 'w') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))

    utils.plot_loss_to_png(os.path.join(target_path,  args.experiment_id + '.json'))
    train_times.append(time.time() - end)

    # Save audio example every 10 epochs
    if epoch%10 == 0:
        x_test, y_test = evaluate.make_an_experiment(model_name=args.model, model_id=args.experiment_id)
        writer.add_audio("x", x_test.permute(1,0).detach().numpy(), epoch, sample_rate=args.sample_rate)
        writer.add_audio("y", y_test.permute(1,0).detach().numpy(), epoch, sample_rate=args.sample_rate)

    if stop:
        print("Apply Early Stopping")
        break