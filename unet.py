print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

import torch
import torch.nn as nn 
import torch.nn.functional as F  
import numpy as np  
from collections import OrderedDict
import os


class Unet(nn.Module):
    '''
    Unet architecture for n2n.
    Weight init through Kaiming method given in paper
    No batch norm, dropout

    '''

    def __init__(self, n=3, m=3):
        super(Unet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(n,48,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48,48,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(48,48,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48,48,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(48,48,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(48,48,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1)
        )

        self.de_conv5 = nn.Sequential(
            nn.Conv2d(96,96,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,1,1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96,96,3,stride=2, padding=1, output_padding=1)
        )

        self.de_conv4 = nn.Sequential(
            nn.Conv2d(96,96,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,1,1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96,96,3,stride=2, padding=1, output_padding=1)
        )

        self.de_conv3 = nn.Sequential(
            nn.Conv2d(96,96,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,1,1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96,96,3,stride=2, padding=1, output_padding=1)
        )

        self.de_conv2 = nn.Sequential(
            nn.Conv2d(96,96,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,1,1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96,96,3,stride=2, padding=1, output_padding=1)
        )

        self.de_conv1 = nn.Sequential(
            nn.Conv2d(96 + n,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,32,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,m,3,1,1),
            nn.ReLU(inplace=True)
        )
        
        
        self.init_weights()

    def init_weights(self):
        '''
        Kaiming method
        '''
        for mod in self.modules():
            if isinstance(mod, nn.ConvTranspose2d) or isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight.data)
                mod.bias.data.zero_()
        

    def summary(self):
        print('Unet summary: ')
        print(self.conv1)
        print(self.conv2)
        print(self.conv3)
        print(self.conv4)
        print(self.conv5)
        print(self.conv6)
        print(self.de_conv5)
        print(self.de_conv4)
        print(self.de_conv3)
        print(self.de_conv2)
        print(self.de_conv1)

    def forward(self,x):
        pool1 = self.conv1(x)
        pool2 = self.conv2(pool1)
        pool3 = self.conv3(pool2)
        pool4 = self.conv4(pool3)
        pool5 = self.conv5(pool4)

        upsample5 = self.conv6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.de_conv5(concat5) 
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.de_conv4(concat4) 
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.de_conv3(concat3) 
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.de_conv2(concat2)   
        concat1 = torch.cat((upsample1, x), dim=1)  
        output = self.de_conv1(concat1)

        return output


