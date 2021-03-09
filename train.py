import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"  # specify which GPU(s) to be used

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
parser.add_argument('--message', type=str, default='layer=8, \
                                                    w=24, \
                                                    num_bins=32, \
                                                    entropy_target=64kbps, \
                                                    alpha=-20, \
                                                    loss weights=[70.0, 1.0, 10.0],\
                                                    summary: No skip')

# Dataset paramaters
parser.add_argument('--root', type=str, default=rootPath, help='root path of dataset')
parser.add_argument('--output', type=str, default="output",
                    help='provide output path base folder name')

# Trainig Parameters
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--num-its', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=16)

# Hyper-parameters
# Quant/Entropy
parser.add_argument('--quant', type=bool, default=False)
parser.add_argument('--quant-active', type=int, default=5)
parser.add_argument('--target-bitrate', type=int, default=64000,
                    help='target bitrate. by default the bitrate of 44.1 mono audio = 705,600')
parser.add_argument('--bitrate-fuzz', type=int, default=450,
                    help='amount of bitrate fuzz allowed around the target bitrate before adjusting tau')

parser.add_argument('--loss-weights', type=list, default=[70.0, 1.0, 10.0],
                    help='weight of each loss term: [mse,quant,entropy]')
parser.add_argument('--num-skips', type=list, default=1)

parser.add_argument('--lr', type=float, default=0.001,
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
                    help='Sequence duration in seconds')
parser.add_argument('--overlap', type=float, default=64)
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
parser.add_argument('--device', action='store_true', default='cuda:0',
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
    total_losses        = utils.AverageMeter()
    mse_losses   = utils.AverageMeter()
    entropy_losses      = utils.AverageMeter()
    quantization_losses = utils.AverageMeter()

    entropy_loss      = torch.Tensor([0]).to(device)
    quantization_loss = torch.Tensor([0]).to(device)
    entropy_avg       = 0
    model.train()

    it_bar = tqdm.tqdm(range(args.num_its), disable=args.quiet, desc='Iteration',position=0)
    batch_bar = tqdm.tqdm(train_sampler, disable=args.quiet, desc='Batch',position=1)
    #torch.autograd.set_detect_anomaly(True)
    # Total num it: num_its x batch_size x training examples
    for it in it_bar:
        #it_bar.set_description("Training Iteration")
        for i, x in enumerate(batch_bar):

            x = x.to(device)
            optimizer.zero_grad()
            y_hat = model(x)

            mse_loss = F.mse_loss(y_hat, x)
            if model.quant_active == True:
                entropy_loss = model.entropy_loss()
                quantization_loss = model.quantization_loss()
                entropy_avg  = model.get_overall_entropy_avg()

            total_loss = ((mse_loss * loss_weights[0]) +\
                          (quantization_loss * loss_weights[1]) +\
                          (entropy_loss * loss_weights[2]))
            total_loss.backward()
            optimizer.step()

            mse_losses.update(mse_loss.item())
            entropy_losses.update(entropy_loss.item())
            quantization_losses.update(quantization_loss.item())
            total_losses.update(total_loss.item(), x.size(1))

        if model.quant_active == True:
            print("entropy loss:",entropy_loss)
            print("quantization loss:",quantization_loss)
            print("mse loss:",mse_loss)
            print("entropy avg:",entropy_avg)
            print("entropy target:",model.target_entropy)
            print("entropy fuzz:",model.entropy_fuzz)

    return mse_losses.avg, entropy_losses.avg, quantization_losses.avg, total_losses.avg, entropy_avg

def valid(args, model, device, valid_sampler, writer, epoch):
    total_losses        = utils.AverageMeter()
    mse_losses   = utils.AverageMeter()
    entropy_losses      = utils.AverageMeter()
    quantization_losses = utils.AverageMeter()

    entropy_loss      = torch.Tensor([0]).to(device)
    quantization_loss = torch.Tensor([0]).to(device)

    model.eval()
    with torch.no_grad():

        for x in valid_sampler:

            x = x.to(device)
            y_hat = model(x)

            mse_loss = F.mse_loss(y_hat, x)
            if model.quant_active == True:
                entropy_loss = model.entropy_loss()
                quantization_loss = model.quantization_loss()

            total_loss = ((mse_loss * loss_weights[0]) +\
                          (quantization_loss * loss_weights[1]) +\
                          (entropy_loss * loss_weights[2]))

            mse_losses.update(mse_loss.item())
            entropy_losses.update(entropy_loss.item())
            quantization_losses.update(quantization_loss.item())
            total_losses.update(total_loss.item(), x.size(1))

        model.entropy_control_update()

    return mse_losses.avg, entropy_losses.avg, quantization_losses.avg, total_losses.avg

use_cuda = torch.cuda.is_available()
device = torch.device(args.device if use_cuda else "cpu")
print("Using GPU:", use_cuda)
#dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}

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
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,#**dataloader_kwargs
)

valid_sampler = torch.utils.data.DataLoader(
    valid_dataset, batch_size=1, drop_last=True#,**dataloader_kwargs
)


model = models.Waveunet(
    n_ch=args.nb_channels,
    model=utils.model_dic[args.model],
    num_skips=args.num_skips)
model.set_network_entropy_target(args.target_bitrate,\
                                args.bitrate_fuzz,\
                                args.sample_rate,\
                                args.seq_dur,\
                                args.overlap)
writer = SummaryWriter(log_dir)
summary(model,(args.nb_channels,args.seq_dur),device='cpu')
print("Model loaded with num. param:", sum(p.numel() for p in model.parameters() if p.requires_grad))
for name, param in model.named_parameters():
    if param.requires_grad and 'quant' in name:
        print(name, param.data)
        
print("entropy target:",model.target_entropy)

model.to(device)
loss_weights = torch.FloatTensor(args.loss_weights).to(device)

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
    print("epoch:",epoch,"\nexperiment:",args.experiment_id,"\nmessage:",args.message,"\bottleneck:",model.bottleneck_dims)
    if epoch == args.quant_active:
        print('quant active')
        model.quant_active = True
        for m in model.skip_encoders:
            m.quant_active = True

    mse_train_loss, ent_train_loss, quant_train_loss, train_loss, entropy_avg = train(args,
                                                                                        model,
                                                                                        device,
                                                                                        train_sampler,
                                                                                        optimizer,
                                                                                        writer,
                                                                                        epoch)
    mse_val_loss, ent_val_loss, quant_valid_loss, valid_loss = valid(args, 
                                                                    model,
                                                                    device,
                                                                    valid_sampler,
                                                                    writer,
                                                                    epoch)
    #scheduler.step(valid_loss)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print('mse_train_loss', mse_train_loss)
    print('mse_val_loss', mse_val_loss)
    print('ent_train_loss', ent_train_loss)
    print('ent_val_loss', ent_val_loss)
    print('quant_train_loss', quant_train_loss)
    print('quant_valid_loss', quant_valid_loss)
    print('train_loss', train_loss)
    print('valid_loss', valid_loss)
    print('entropy_avg', entropy_avg)
    print('quant_bins', model.state_dict()['quant_bins'].clone().cpu().data.numpy())
    print('quant_alpha', model.state_dict()['quant_alpha'].clone().cpu().data)

    writer.add_scalar("mse_train_loss", mse_train_loss, epoch)
    writer.add_scalar("mse_valid_loss", mse_val_loss, epoch)
    writer.add_scalar("entropy_train_loss", ent_train_loss, epoch)
    writer.add_scalar("entropy_valid_loss", ent_val_loss, epoch)
    writer.add_scalar("quant_train_loss", quant_train_loss, epoch)
    writer.add_scalar("quant_valid_loss", quant_valid_loss, epoch)
    writer.add_scalar("total_train_loss", train_loss, epoch)
    writer.add_scalar("total_valid_loss", valid_loss, epoch)
    writer.add_scalar("entropy_avg", entropy_avg, epoch)
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
    # if epoch%10 == 0:
    #     x_test, y_test = evaluate.make_an_experiment(model_name=args.model, model_id=args.experiment_id)
    #     writer.add_audio("x", x_test.permute(1,0).detach().numpy(), epoch, sample_rate=args.sample_rate)
    #     writer.add_audio("y", y_test.permute(1,0).detach().numpy(), epoch, sample_rate=args.sample_rate)

    if stop:
        print("Apply Early Stopping")
        break