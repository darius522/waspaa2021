import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"  # specify which GPU(s) to be used

from sacred import Experiment
from config import config_ingredient

import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
import tqdm
import json
import csv

import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import torchaudio

import librosa
import librosa.display

import time
import IPython.display as ipd

import soundfile as sf
import logging

import models
import data
import utils
import evaluate

ex = Experiment('HARP Training', ingredients=[config_ingredient])

@ex.config
def set_seed():
    seed = 1337

##############################################################
##################### Train Routines #########################
##############################################################

@config_ingredient.capture
def train(config, model, device, train_sampler, optimizer, writer, epoch):

    loss_weights = torch.FloatTensor(config['loss_weights']).to(device)

    total_losses        = utils.AverageMeter()
    mse_losses   = utils.AverageMeter()
    entropy_losses      = utils.AverageMeter()
    quantization_losses = utils.AverageMeter()

    entropy_loss      = torch.Tensor([0]).to(device)
    quantization_loss = torch.Tensor([0]).to(device)
    entropy_avg       = 0
    model.train()

    it_bar = tqdm.tqdm(range(config['num_its']), disable=config['quiet'], desc='Iteration',position=0)
    batch_bar = tqdm.tqdm(train_sampler, disable=config['quiet'], desc='Batch',position=1)


    for it in it_bar:
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

@config_ingredient.capture
def valid(config, model, device, valid_sampler, writer, epoch):

    loss_weights = torch.FloatTensor(config['loss_weights']).to(device)

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

        if model.quant_active == True:
            model.entropy_control_update()

    return mse_losses.avg, entropy_losses.avg, quantization_losses.avg, total_losses.avg

@config_ingredient.capture
def load_a_model(config):

    if 'baseline' in config['model']:
        model = models.Waveunet(n_ch=config['nb_channels'], 
                                num_layers=config['num_layers'], 
                                tau_change=config['tau_change'], 
                                quant_alpha=config['quant_alpha'])
    elif 'harp' in config['model']:
        model = models.HARPNet(n_ch=config['nb_channels'], 
                                num_skips=config['num_skips'],
                                tau_change=config['tau_change'], 
                                quant_alpha=config['quant_alpha'])
        
    model.set_network_entropy_target(config['target_bitrate'],
                                    config['bitrate_fuzz'],
                                    config['sample_rate'],
                                    config['seq_dur'],
                                    config['overlap'])
    
    summary(model,(config['nb_channels'],config['seq_dur']),device='cpu')
    model.to(config['device'])
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model loaded with num. param:", num_param)
    print("Model's entropy target set to:",model.target_entropy)

    return model, num_param

@ex.automain
def main(cfg):

    config = cfg['config']

    use_cuda = torch.cuda.is_available()
    device = torch.device(config['device'] if use_cuda else "cpu")
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {'num_workers': 0} if use_cuda else {}
    torch.set_num_threads(1)

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])

    ##############################################################
    ################# Datasets / Dataloaders #####################
    ##############################################################

    allPaths  = []
    allPaths += [os.path.join(config['root'],song) for song in os.listdir(config['root']) if not os.path.isdir(os.path.join(config['root'], song))]
    totLen    = len(allPaths)
    random.seed(0)

    with open('./data/test_set.csv', newline='') as f:
        reader  = csv.reader(f)
        tePaths = list(reader)

    allPaths = [p for p in allPaths if p not in tePaths]
    random.shuffle(allPaths)
    trPaths = allPaths[:np.int(totLen*.9)]
    vPaths  = allPaths[np.int(totLen*.9):]

    random.shuffle(trPaths)
    tic=time.time()

    train_dataset, valid_dataset = data.load_datasets(config, train=trPaths, valid=vPaths)

    # create output dir / log dir if not exist
    model_path = Path(os.path.join(config['output_dir'], config['model']))
    if not model_path.exists: model_path.mkdir(parents=True, exist_ok=True)
    target_path = Path(os.path.join(model_path, config['model_id']))
    target_path.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(target_path,'log_dir'))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    utils.dataset_items_to_csv(path=os.path.join(target_path,'train_set.csv'),items=trPaths)
    utils.dataset_items_to_csv(path=os.path.join(target_path,'val_set.csv'),items=vPaths)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True,#**dataloader_kwargs
    )

    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, drop_last=True#,**dataloader_kwargs
    )

    ##############################################################
    ########################## Model #############################
    ##############################################################

    model, num_param = load_a_model()

    allPaths  = []
    allPaths += [os.path.join(config['root'],song) for song in os.listdir(config['root']) if not os.path.isdir(os.path.join(config['root'], song))]
    totLen    = len(allPaths)
    random.seed(0)

    random.shuffle(allPaths)
    trPaths = allPaths[:np.int(totLen*.9)]
    vPaths  = allPaths[np.int(totLen*.9):np.int(totLen*.95)]
    tePaths = allPaths[np.int(totLen*.95):]

    random.shuffle(trPaths)
    tic=time.time()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr']
    )

    es = utils.EarlyStopping(patience=config['patience'])

    t = tqdm.trange(1, config['epochs'] + 1, disable=config['quiet'])
    train_losses = []
    valid_losses = []
    train_times  = []
    best_epoch   = 0

    ##############################################################
    ######################### Training ###########################
    ##############################################################

    for epoch in t:

        t.set_description("Training Epoch")
        end = time.time()
        print("epoch:",epoch,"\nexperiment:",config['model_id'])
        if epoch == config['quant_active']:
            print('quant active')
            model.quant_active = True
            for m in model.skip_encoders:
                m.quant_active = True

        mse_train_loss, ent_train_loss, quant_train_loss, train_loss, entropy_avg = train(config,
                                                                                            model,
                                                                                            device,
                                                                                            train_sampler,
                                                                                            optimizer,
                                                                                            writer,
                                                                                            epoch)
        mse_val_loss, ent_val_loss, quant_valid_loss, valid_loss = valid(config, 
                                                                        model,
                                                                        device,
                                                                        valid_sampler,
                                                                        writer,
                                                                        epoch)
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
        writer.add_histogram('quantization_bins', model.state_dict()['quant_bins'].clone().cpu().data.numpy(), epoch)
        writer.add_scalar('quantization_alpha', model.state_dict()['quant_alpha'].clone().cpu().data, epoch)

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
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=config['model_id']
        )

        # save params
        params = {
            'model_num_param': num_param,
            'epochs_trained': epoch,
            'config': config,
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs
        }

        # Post epoch business
        with open(Path(target_path,  config['model_id'] + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        utils.plot_loss_to_png(os.path.join(target_path,  config['model_id'] + '.json'))
        train_times.append(time.time() - end)

        # Evaluate SNR every 10 epochs
        if epoch%10 == 0:
            snr = evaluate.evaluate(config=config)
            writer.add_scalar('snr', snr, epoch)

        if stop:
            print("Apply Early Stopping")
            break