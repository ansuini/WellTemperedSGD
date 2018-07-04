import torch
import torch.nn as nn
import torch.nn.functional as F

# LeNet example architecture (no dropout)
class LeNet(nn.Module):
    '''
    The mythical LeNet network by Yann LeCun
    '''
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def extract(self, x):
        out1 = F.relu(F.max_pool2d(self.conv1(x),    2 ) )
        out2 = F.relu(F.max_pool2d(self.conv2(out1), 2 ) )       
        t = out2.view(-1, 320)
        out3 = F.relu(self.fc1(t))
        t = self.fc2(out3)
        out4 = F.log_softmax(t, dim=1)        
        return out1, out2, out3, out4

class Net(nn.Module):
    '''
    A larger version of the mythical LeNet network by Yann LeCun
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d( self.conv1(x), 2 ) )
        x = F.relu(F.max_pool2d( self.conv2(x), 2 ) )      
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def extract(self, x):
        out1 = F.relu(F.max_pool2d(self.conv1(x),    2 ) )
        out2 = F.relu(F.max_pool2d(self.conv2(out1), 2 ) )       
        t = out2.view(-1, 1600)
        out3 = F.relu(self.fc1(t))
        t = self.fc2(out3)
        out4 = F.log_softmax(t, dim=1)        
        return out1, out2, out3, out4