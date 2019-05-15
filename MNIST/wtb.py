#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from matplotlib import pyplot as plt

import numpy as np
import os
from os.path import join

ROOT='/home/ansuini/repos/WellTemperedSGD/MNIST'
RES=join(ROOT,'results')
datum='data_shuffled'

#-----------------------------------------------------------------------------------------
def init_tensors(model,verbose=False):
    '''
    Init to zero a list of tensors with the same shapes of model.parameters()
    '''
    tensors = [torch.zeros_like(p) for p in model.parameters()]     
    
    if verbose:
        print('Tensors shapes:')
        _ = [print(t.shape) for t in tensors]
    
    return tensors

def init_tensors_one(model,verbose=False):
    '''
    Init to one a list of tensors with the same shapes of model.parameters()
    '''
    tensors = [torch.ones_like(p) for p in model.parameters()]     
    
    if verbose:
        print('Tensors shapes:')
        _ = [print(t.shape) for t in tensors]
    
    return tensors

def acc_grad(grad, model):
    '''
    Accumulate grad in a list of tensors 
    of the same structure of model.parameters() 
    '''
    for g, p in zip(grad, model.parameters()):
        g += p.grad
    return grad
        
def acc_grad2(grad2, model):
    '''
    Accumulate squared grad in a list of tensors 
    of the same structure of model.parameters() 
    '''
    for g, p in zip(grad2, model.parameters()):
        g += torch.mul(p.grad, p.grad)
    return grad2

def clone_tensors(tensors):
    '''
    Clone gradient data to make some tests
    '''
    return [t.grad.clone() for t in tensors]

def compute_snr(grad, grad2, B, device):
    '''
    Compute snr
    '''  
    
    epsilon = 1e-8 #small quantity to be added to err in order to avoid division by zero
    
    snr = [] #list of tensors with the same structure as model.parameters()
    
    for g, g2 in zip(grad, grad2):
        
        # work with clones in order to avoid modifications of the original data in this function
        g_copy  = g.clone()
        g2_copy = g2.clone()
    
        # average over number of batches (B is the same as in the paper)
        g_copy = g_copy/B   
        g2_copy = g2_copy/B  
        
        # compute error    
        assert(torch.sum(g2_copy - g_copy*g_copy >= 0) ) # assert if the variance is non-negative
        
        err = torch.sqrt( ( g2_copy - g_copy*g_copy )/ B ) # the error is the square root of the variance divided by B
        err[err==0] = epsilon # add small positive quantity if err is 0
        
        # compute signal to error ratio
        # snr is the ratio between the abs value of the gradient averaged
        # over B batches and the err
        snr.append(torch.div( torch.abs(g_copy), err ) ) 
            
    return snr

#--------------------------------------------------------------------------------------------
class Net(nn.Module):
    # dropout is 0 by default
    def __init__(self,p1=0.0, p2=0.0):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout1 = nn.Dropout(p=p1)
        self.dropout2 = nn.Dropout(p=p2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#-----------------------------------------------------------------------------------------
def stats(model, loader, device):    
    model.eval()    
    loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
                
            data, target = data.to(device), target.to(device)
            output = model(data)            
            loss += F.nll_loss(output, target)*data.shape[0]
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
                
    loss /= len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)

    if loader.dataset.train==True:
        datatype='training'
    else:
        datatype='test'
        
    print(datatype + ' set: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
           loss, correct, len(loader.dataset), acc))
    
    return loss.item(),acc


def main():

    #------------------------------------ Arguments --------------------------------------------
    parser = argparse.ArgumentParser(description='PyTorch MNIST with WTB')
    
    parser.add_argument('--wtb', type=int, default=0,
                        help='use wtb (default: no 0, yes 1)')
    
    parser.add_argument('--p1', type=float, default=0.0, metavar='dropout',
                        help='first dropout level (default: 0.0)')
    
    parser.add_argument('--p2', type=float, default=0.0, metavar='dropout',
                        help='second dropout level (default: 0.0)')
    
    parser.add_argument('--batch_size', type=int, default=6000, metavar='N',
                        help='input batch size for training (default: 6000)')
    
    parser.add_argument('--nsamples_train', type=int, default=60000, metavar='N',
                        help='input number of samples used for training (default: 60000)')
    
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--save', type=int, default=0,
                        help='for saving the current model and training info')
    
    args = parser.parse_args()
    print('Args: {}'.format(args))
    print('Data: {}'.format(join(ROOT,datum)))

     
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    #-------------------------------------- Data -----------------------------------------------
    # Please notice that shuffle is False here in the training_loader. 
    # This is essential if we want to restrict the training dataset
    # to nsamples_training < 60000. It is not essential to set it to 
    # False in the test_loader

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(join(ROOT,datum), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)


    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(join(ROOT,datum), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    
    
    #--------------------------------- Model + optimizer ------------------------------------
    model = Net().to(device)
    print(summary(model,(1,28,28)))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    #---------------------------------- Training --------------------------------------------
    if args.wtb:
        print('Well tempered backprop!')
    else:
        print('Normal backprop!')

    train_stats = []
    test_stats = []
    if args.wtb:
        fraction = []

    # init snr to 1 the first time
    snr = init_tensors_one(model)

    # compute total number of parameters
    tot = 0
    for p in model.parameters():
        tot += p.numel()
    
    # iterate over epochs
    for epoch in range(1, args.epochs + 1):
        print('\nEpoch: {}'.format(epoch))      
        
        # init tensors to store gradients and gradients squared 
        grad = init_tensors(model)
        grad2 = init_tensors(model)

        model.train()
        B = 0 # count mini-batches  
        
        # iterate on current epoch accumulating grad and grad2
        for batch_idx, (data, target) in enumerate(train_loader):

            # control the number of training samples in case you want to train on a subset of the data
            if batch_idx*train_loader.batch_size > args.nsamples_train:
                break 
                
            B += 1        
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # accumulate grads
            grad = acc_grad(grad,model)
            grad2 = acc_grad2(grad2,model)

            # gradient modification and update
            with torch.no_grad():
                for p,s in zip(model.parameters(),snr): 

                    # modify grad with snr computed on the previous epoch
                    p.grad = torch.where( s < 1, s*p.grad, p.grad) 

                    # update parameters with the new gradient
                    p.data -= args.lr*p.grad.data

        # if wtb compute snr, otherwise snr will remain 1 and will not affect backprop
        if args.wtb:
            # update snr at the end of the epoch
            with torch.no_grad():
                snr = compute_snr(grad, grad2, B, device)

                # compute fraction of parameters whose snr is < 1
                count = 0  
                for s in snr:
                    count += (s < 1).sum().item()
                fraction.append(count/tot)
                print('Fraction (global): {}'.format(count/tot))  
                
        train_stats.append(stats(model, train_loader, device))
        test_stats.append(stats(model, test_loader, device))
        
        # save model weigths at each epoch
        if args.save:
            if args.wtb:
                torch.save(model.state_dict(), join(RES, 'model_wtb_' + str(epoch)))
            else:
                torch.save(model.state_dict(), join(RES, 'model_norm_' + str(epoch)))

    # after all epochs
    if args.save:
        if args.wtb:
            np.save(join(RES, 'train_stats_wtb'), train_stats)
            np.save(join(RES, 'test_stats_wtb'), test_stats)
            np.save(join(RES, 'fraction'), fraction)
        else:
            np.save(join(RES, 'train_stats_norm'), train_stats)
            np.save(join(RES, 'test_stats_norm'), test_stats)
        
if __name__ == '__main__':
    main()