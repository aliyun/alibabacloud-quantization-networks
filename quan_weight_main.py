#!/usr/bin/env python
# -*- coding: utf-8 -*-
# quan_weight_main.py is used to train the weight quantized model.

from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import numpy as np
import sys
import time
import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.nn.parameter import Parameter
from torchvision import transforms as T 

from config import *
import models
from data_pre import Lighting, Preprocessor
from utils import Logger, AverageMeter
from utils import load_checkpoint, save_checkpoint
from utils import RandomResized
from anybit import QuaOp
from evaluators import accuracy
import pdb

# define global qua_op
qua_op = None

def get_data(split_id, data_dir, img_size, scale_size, batch_size,
             workers, train_list, val_list):
    root = data_dir

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # RGB imagenet
    # with data augmentation 
    train_transformer = T.Compose([
        T.Resize(scale_size),
        T.RandomCrop(img_size),
        #T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),   # [0, 255] to [0.0, 1.0]
        normalizer,     #  normalize each channel of the input
     ])

    test_transformer = T.Compose([
        T.Resize(scale_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_list, root=root,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomSampler(train_list),
        pin_memory=True, drop_last=False)

    val_loader = DataLoader(
        Preprocessor(val_list, root=root,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return train_loader, val_loader

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    data_dir = osp.join(args.data_dir, args.dataset)
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        sys.stdout = Logger(osp.join(args.logs_dir, 'evaluate-log.txt'))
    print('\n################## setting ###################')
    print(parser.parse_args())
    print('################## setting ###################\n')
    # Create data loaders
    def readlist(fpath):
        lines=[]
        with open(fpath, 'r') as f:
            data = f.readlines()

        for line in data:
            name, label = line.split()
            lines.append((name, int(label)))
        return lines

    # Load data list
    if osp.exists(osp.join(data_dir, 'train.txt')):
        train_list = readlist(osp.join(data_dir, 'train.txt'))
    else:
        raise RuntimeError("The training list -- {} doesn't exist".format(train_list))

    if osp.exists(osp.join(data_dir, 'val.txt')):
        val_list = readlist(osp.join(data_dir, 'val.txt'))
    else:
        raise RuntimeError("The val list -- {} doesn't exist".format(val_list))


    if args.scale_size is None :
        args.scale_size = 256 
    if args.img_size is None :
        args.img_size = 224 

    train_loader, val_loader = \
        get_data(args.split, data_dir, args.img_size,
                 args.scale_size, args.batch_size, args.workers,
                 train_list, val_list)
    # Create model
    #num_classes = 1000 # imagenet 1000
    model = models.create(args.arch, False, num_classes=1000)
    # create alpha and belta
    count = 0
    for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count = count + 1
    alpha = []
    beta = []
    for i in range(count-2):
        alpha.append(Variable(torch.FloatTensor([0.0]).cuda(), requires_grad=True))
        beta.append(Variable(torch.FloatTensor([0.0]).cuda(), requires_grad=True))

    if args.adam:
        print('The optimizer is Adam !!!')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        optimizer_alpha = torch.optim.Adam(alpha, lr=args.lr)
        optimizer_beta = torch.optim.Adam(beta, lr=args.lr)
    else:
        print('The optimizer is SGD !!!')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        optimizer_alpha = torch.optim.SGD(alpha, lr=args.lr,
                                 momentum=args.momentum)
        optimizer_beta = torch.optim.SGD(beta, lr=args.lr,
                                 momentum=args.momentum)
    # model Load from checkpoint
    start_epoch = best_top1 = 0
    if args.pretrained:
        print('=> Start load params from pre-trained model...')
        checkpoint = load_checkpoint(args.pretrained)
        if 'alexnet' in args.arch or 'resnet' in args.arch:
            model.load_state_dict(checkpoint)
        else:
            raise RuntimeError('The arch is ERROR!!!')
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_alpha.load_state_dict(checkpoint['optimizer_alpha'])
        optimizer_beta.load_state_dict(checkpoint['optimizer_beta'])
        alpha = checkpoint['alpha']
        beta = checkpoint['beta']
        start_epoch = args.resume_epoch
        print("=> Finetune Start epoch {} "
              .format(start_epoch))

    
    model = nn.DataParallel(model).cuda()  

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    qw_values = QW_values[args.offline_biases]
    print('qw_values: ', qw_values)
    global qua_op
    qua_op = QuaOp(model, QW_biases[args.offline_biases], QW_values=qw_values)

    evaluator = Evaluator(model, criterion, alpha, beta)
    if args.evaluate:
        print('Test model: \n')
        evaluator.evaluate(val_loader, T=1)
        return

    # Trainer
    trainer = Trainer(model, criterion, alpha, beta)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = args.step_size
        decay_step = args.decay_step
        lr = args.lr if epoch < step_size else \
             args.lr * (0.1 ** ((epoch - step_size) // decay_step + 1))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    trainer.show_info(with_arch=True, with_grad=False)
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        t = (epoch + 1) * args.temperature      # linear
        print('W_T = ', t)
         
        trainer.train(epoch, train_loader, optimizer, optimizer_alpha, 
                      optimizer_beta, T=t, print_info=args.print_info)
        if epoch < args.start_save:
            continue
        top1 = evaluator.evaluate(val_loader, T=t)
    
        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
                        'state_dict':model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'optimizer_alpha': optimizer_alpha.state_dict(),
                        'optimizer_beta': optimizer_beta.state_dict(),
                        'alpha': alpha,
                        'beta': beta},
                        is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.2%}  model_best: {:5.2%} \n'.
              format(epoch, top1, best_top1))

        if (epoch+1) % 5 == 0:
            model_name = 'epoch_'+ str(epoch) + '.pth.tar'
            torch.save({'state_dict':model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'optimizer_alpha': optimizer_alpha.state_dict(),
                        'optimizer_beta': optimizer_beta.state_dict(),
                        'alpha': alpha,
                        'beta': beta},
                        osp.join(args.logs_dir, model_name))

class Trainer(object):
    def __init__(self, model, criterion, alpha, beta):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.alpha = alpha
        self.beta = beta
        self.init = False

    def train(self, epoch, data_loader, optimizer, optimizer_alpha, 
              optimizer_beta, T=1, print_freq=1, print_info=10):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            if epoch == 0 and i == 0:
                self.init = True
            else:
                self.init = False            
            data_time.update(time.time() - end)

            inputs_var, targets_var = self._parse_data(inputs)
            
            qua_op.quantization(T, self.alpha, self.beta, init=self.init)
    
            loss, prec1, prec5 = self._forward(inputs_var, targets_var)
            losses.update(loss.data[0], targets_var.size(0))
            top1.update(prec1, targets_var.size(0))
            top5.update(prec5, targets_var.size(0))

            optimizer.zero_grad()
            optimizer_alpha.zero_grad()
            optimizer_beta.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 5.0)

            qua_op.restore_params()
            alpha_grad, beta_grad = qua_op.updateQuaGradWeight(T, self.alpha, self.beta, init=self.init)
            for index in range(len(self.alpha)):
                self.alpha[index].grad = Variable(torch.FloatTensor([alpha_grad[index]]).cuda())
                self.beta[index].grad = Variable(torch.FloatTensor([beta_grad[index]]).cuda())

            optimizer.step()
            optimizer_alpha.step()
            optimizer_beta.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec@1 {:.2%} ({:.2%})\t'
                      'Prec@5 {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              top1.val, top1.avg,
                              top5.val, top5.avg))
        if (epoch+1) % print_info == 0:
            self.show_info()

    def show_info(self, with_arch=False, with_grad=True):
        if with_arch:
            print('\n\n################# model modules ###################')
            for name, m in self.model.named_modules():
                print('{}: {}'.format(name, m))
            print('################# model modules ###################\n\n')

        if with_grad:
            print('################# model params diff ###################')
            for name, param in self.model.named_parameters():
                mean_value = torch.abs(param.data).mean()
                mean_grad = torch.abs(param.grad).mean().data[0] + 1e-8
                print('{}: size{}, data_abd_avg: {}, dgrad_abd_avg: {}, data/grad: {}'.format(name,
                                                param.size(), mean_value, mean_grad, mean_value/mean_grad))
            print('################# model params diff ###################\n\n')

        else:
            print('################# model params ###################')
            for name, param in self.model.named_parameters():
                print('{}: size{}, abs_avg: {}'.format(name,
                                                       param.size(),
                                                       torch.abs(param.data.cpu()).mean()))
            print('################# model params ###################\n\n')

    def _parse_data(self, inputs):
        imgs, _, labels = inputs
        inputs_var = [Variable(imgs)]
        targets_var = Variable(labels.cuda())
        return inputs_var, targets_var

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec1, prec5= accuracy(outputs.data, targets.data, topk=(1,5))
            prec1 = prec1[0]
            prec5 = prec5[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec1, prec5

class Evaluator(object):
    def __init__(self, model, criterion, alpha, beta):
        super(Evaluator, self).__init__()
        self.model = model
        self.criterion = criterion
        self.alpha = alpha
        self.beta = beta 

    def evaluate(self, data_loader, T=1, print_freq=1):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()

        end = time.time()
        print('alpha: ', self.alpha)
        print('beta: ', self.beta)
        qua_op.quantization(T, self.alpha, self.beta, init=False, train_phase=False)

        for i, inputs in enumerate(data_loader):
            inputs_var, targets_var = self._parse_data(inputs)

            loss, prec1, prec5 = self._forward(inputs_var, targets_var)

            losses.update(loss.data[0], targets_var.size(0))
            top1.update(prec1, targets_var.size(0))
            top5.update(prec5, targets_var.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.4f} ({:.4f})\t'
                      'Prec@1 {:.2%} ({:.2%})\t'
                      'Prec@5 {:.2%} ({:.2%})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              top1.val, top1.avg,
                              top5.val, top5.avg))

        qua_op.restore_params()

        print(' * Prec@1 {:.2%} Prec@5 {:.2%}'.format(top1.avg, top5.avg))

        return top1.avg

    def _parse_data(self, inputs):
        imgs, _, labels = inputs
        inputs_var = [Variable(imgs, volatile=True)]
        targets_var = Variable(labels.cuda(), volatile=True)
        return inputs_var, targets_var

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec1, prec5= accuracy(outputs.data, targets.data, topk=(1,5))
            prec1 = prec1[0]
            prec5 = prec5[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec1, prec5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='imagenet')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--scale_size', type=int, default=256,
                        help="val resize image size, default: 256 for ImageNet")
    parser.add_argument('--img_size', type=int, default=224,
                        help="input image size, default: 224 for ImageNet")
    # model
    parser.add_argument('-a', '--arch', type=str, default='alexnet',
                        choices=models.names())
    # optimizer
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--step_size', type=int, default=25)
    parser.add_argument('--decay_step', type=int, default=25)
    
    # training configs  pretrained_model
    parser.add_argument('--pretrained', type=str, default='', metavar='PATH')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--resume_epoch', type=int,default=0)
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--adam', action='store_true',
                        help="use Adam")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--print-info', type=int, default=10)
    parser.add_argument('--temperature', type=int, default=10)
    parser.add_argument('--offline_biases', type=str, default='')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())

