from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import os
from os.path import join

ROOT='/home/ansuini/repos/WellTemperedSGD/MNIST'
RES=join(ROOT,'results')
datum='data_shuffled'

class Net(nn.Module):
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
    
    
    
def stats(model, loader, device):    
    model.eval()    
    loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
                
            data, target = data.to(device), target.to(device)
            output = model(data)            
            loss += F.nll_loss(output, target)
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
    
    

def train(model, train_loader, nsamples, optimizer, epoch, device):
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        #print(batch_idx*train_loader.batch_size)
        if batch_idx*train_loader.batch_size > nsamples:
            #print('Done.')
            break
            
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--p1', type=float, default=0.0, metavar='dropout',
                        help='first dropout level (default: 0.0)')
    
    parser.add_argument('--p2', type=float, default=0.0, metavar='dropout',
                        help='second dropout level (default: 0.0)')
    
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--nsamples_train', type=int, default=60000, metavar='N',
                        help='input number of samples used for training (default: 60000)')
    
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    
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
    
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save', type=int, default=0,
                        help='for saving the current model and training info')
    
    
    args = parser.parse_args()
    print('Args: {}'.format(args))
    print('Data: {}'.format(join(ROOT,datum)))
   
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(join(ROOT,datum), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(join(ROOT,datum), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net(args.p1, args.p2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    
    # training
    train_stats = []
    test_stats = []
    
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, args.nsamples_train, optimizer, epoch, device)
        print('\nEpoch: {}'.format(epoch))
        train_stats.append(stats(model, train_loader, device))
        test_stats.append(stats(model, test_loader, device))

    if args.save:
        print('Saving model.')
        torch.save(model.state_dict(),
                   join(RES,'mnist_cnn_' + str(args.p2) + '.pt') )
        
        torch.save(train_stats, 
                   join(RES,'train_stats_' + str(args.p2) + '.p') )
        
        torch.save(test_stats, 
                   join(RES,'test_stats_' + str(args.p2) + '.p') )
        
    print('Done.')
    
        
if __name__ == '__main__':
    main()