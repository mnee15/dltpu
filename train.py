from importlib import import_module
from pathlib import Path
import os
import random
from math import pi
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets.Dataset import PhaseDataset
from loss import create_criterion
from scheduler import create_scheduler

from optimizer import create_optimizer

import wandb
import yaml
from easydict import EasyDict

import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def save_model(model, saved_dir, args):
    file_name = args.experiment_name + '.pth'
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)


def train(args):
    print(f'Start training...')

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- data_set
    train_dataset = PhaseDataset(dataset_path=args.dataset_path, mode='train', low=args.low, fom=args.fom)
    val_dataset = PhaseDataset(dataset_path=args.dataset_path, mode='val', low=args.low, fom=args.fom)

    # -- datalodaer
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=args.valid_batch_size,
                            shuffle=False,
                            num_workers=2,
                            drop_last=False)
                                         
    # -- model
    model_module = getattr(import_module('models.' + args.model_name), args.model_name)
    model = model_module()

    if args.resume:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        
    # device
    model = model.to(device)  

    # -- loss & metric
    criterion1 = create_criterion(args.criterions[0])
    criterion2 = create_criterion(args.criterions[1])
    
    # -- optimizer
    optimizer = create_optimizer(optimizer_name=args.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5, momentum=0.9)

    # -- scheduler
    if args.scheduler:
        scheduler = create_scheduler(optimizer, args.scheduler, args.epochs, args.lr)

    # -- train 
    best_loss = np.inf
    val_every = 1
    
    # Grad accumulation
    NUM_ACCUM = args.grad_accum
    optimizer.zero_grad()
    
    # Early Stopping
    PATIENCE = args.patience
    counter = 0

    # fp16
    scaler = torch.cuda.amp.GradScaler()

    # start train
    for epoch in range(args.epochs):
        print('\n')
        print(f'Epoch : {epoch + 1}')
        model.train()

        with tqdm(total=len(train_loader)) as pbar:
            for step, (phase, gt, mask) in enumerate(train_loader):          
                phase, gt, mask = phase.to(device), gt.to(device), mask.to(device)

                # inference
                with torch.cuda.amp.autocast():
                    pred = model(phase) * mask
                    loss1 = criterion1(pred, gt)
                    loss2 = criterion2(pred, gt)
                    loss = loss1 * args.criterion_weights[0] + loss2 * args.criterion_weights[1]

                scaler.scale(loss).backward()

                if step % NUM_ACCUM == 0:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                
                scaler.update()
                pbar.update(1)
                
                logging = {
                    'Tr loss': round(loss.item(), 4),
                    'Tr loss 1': round(loss1.item(), 4),
                    'Tr loss 2': round(loss2.item(), 4)
                }
                pbar.set_postfix(logging)

                if (step + 1) % args.log_interval == 0:
                    current_lr = get_lr(optimizer)
                    logging['lr'] = current_lr
                    # wandb
                    if args.log_wandb:
                        wandb.log(logging)
            
        # validation / save best model
        if (epoch + 1) % val_every == 0:
            avrg_loss, avrg_loss1, avrg_loss2 = validation(model, val_loader, device, criterion1, criterion2, epoch, args)
            if avrg_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_loss = avrg_loss
                save_model(model, saved_dir, args)
                counter = 0
            else:
                counter += 1

            # wandb
            if args.log_wandb:
                wandb.log(
                    {
                        'Val Loss': avrg_loss,
                        'Val Loss 1' : avrg_loss1,
                        'Val Loss 2' : avrg_loss2
                    }
                )

            if (args.early_stopping) and (counter > PATIENCE):
                print('Early Stopping...')
                break
        
        if args.scheduler:
            if args.scheduler == 'ReduceOP':
                scheduler.step(avrg_loss)
            else:
                scheduler.step()


def validation(model, data_loader, device, criterion1, criterion2, epoch, args):
    print(f'Start validation!')
    model.eval()
    total_loss1, total_loss2, cnt = 0, 0, 0

    with torch.no_grad():        
        for step, (phase, gt, mask) in enumerate(tqdm(data_loader)):      
            phase, gt, mask = phase.to(device), gt.to(device), mask.to(device)       
            
            with torch.cuda.amp.autocast():
                pred = model(phase) * mask
                loss1 = criterion1(pred, gt)
                loss2 = criterion2(pred, gt)

                total_loss1 += loss1
                total_loss2 += loss2
                cnt += 1

        avrg_loss1 = total_loss1 / cnt
        avrg_loss2 = total_loss2 / cnt
        avrg_loss = avrg_loss1 * args.criterion_weights[0] + avrg_loss2 * args.criterion_weights[1]
        print(f'Validation #{epoch + 1} || Loss: {round(avrg_loss.item(), 4)} || Loss 1 : {round(avrg_loss1.item(), 4)} || Loss 2: {round(avrg_loss2.item(), 4)}')
        
    return avrg_loss, avrg_loss1, avrg_loss2


if __name__ == "__main__":
    CONFIG_FILE_NAME = "./config/config.yaml"
    with open(CONFIG_FILE_NAME, "r") as yml_config_file:
        args = yaml.load(yml_config_file, Loader=yaml.FullLoader)
        args = EasyDict(args['train'])

    print(args)
    seed_everything(args.seed)

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    CFG = {
        "epochs" : args.epochs,
        "batch_size" : args.batch_size,
        "learning_rate" : args.lr,
        "seed" : args.seed,
        "optimizer" : args.optimizer,
        "scheduler" : args.scheduler,
        "criterion1" : args.criterions[0],
        "criterion2" : args.criterions[1]
    }

    if args.log_wandb:
        wandb.init(
            project=args.project, entity=args.entity, name=args.experiment_name, config=CFG,
        )

        wandb.define_metric("Tr Loss", summary="min")
        wandb.define_metric("Tr Loss 1", summary="min")
        wandb.define_metric("Tr Loss 2", summary="min")
        wandb.define_metric("Val Loss", summary="min")
        wandb.define_metric("Val Loss 1", summary="min")
        wandb.define_metric("Val Loss 2", summary="min")

    train(args)

    if args.log_wandb:
        wandb.finish()

