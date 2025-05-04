import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import torch.autograd as autograd
 
import numpy as np
import math
import random
from torch.nn import BatchNorm2d, BatchNorm1d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm,InstanceNorm2d, Dropout2d
import util
from utils import MSTGCN, Dynamic_Delay_Aware_Module


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def rand_mask(x,alpha,self):
    rand1=torch.rand(x.shape).to(x.device)
    mask=(rand1>alpha).int().float()
    x=x*mask
    return x


class Decoder(nn.Module):
    def __init__(self,  num_nodes, k_num, l, length=12, residual_channels=32, dilation_channels=32, K=3, Kt=3):
        super(Decoder, self).__init__()
        self.l = l
        print(l)
        self.block = nn.ModuleList()
        for i in range(l - 1):
            self.block.append(MSTGCN(dilation_channels, dilation_channels, num_nodes, length, K, Kt))
        self.DDM_decoder = Dynamic_Delay_Aware_Module(k_num, dilation_channels, length)

    def forward(self, v, Trend_Matrix, A):
        v = self.DDM_decoder(v)
        for i in range(self.l - 1):
            v = self.block[i](v, Trend_Matrix, A)
        return v


class Encoder(nn.Module):
    def __init__(self, num_nodes, k_num,l, length=12,
                 in_dim=1, residual_channels=32, dilation_channels=32, K=3, Kt=3):
        super(Encoder, self).__init__()
        self.l = l
        print(l)
        self.block1 = nn.ModuleList()
        self.block1.append(MSTGCN(in_dim, dilation_channels, num_nodes, length, K, Kt))
        self.block1.append(MSTGCN(dilation_channels, dilation_channels, num_nodes, length, K, Kt))
        self.DDM_encoder = Dynamic_Delay_Aware_Module(k_num, in_dim, length)

    def forward(self, v, Trend_Matrix, A):
        v = self.DDM_encoder(v)
        for i in range(self.l - 1):
            v = self.block1[i](v, Trend_Matrix, A)
        return v


class DDAMGCN(nn.Module):
    def __init__(self, device, num_nodes,k_num, l, dropout=0.5, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3):
        super(DDAMGCN, self).__init__()
        print(device, num_nodes,k_num, l, dropout,length,
                 in_dim, out_dim, residual_channels, dilation_channels,
                 skip_channels, end_channels, kernel_size, K, Kt)
        tem_size = length
        self.num_nodes = num_nodes
        self.dropout=0.5
        self.l = l
        self.in_dim=in_dim

        self.conv1 = Conv2d(dilation_channels, 12, kernel_size=(1, tem_size), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.supports = supports
        self.encoder = Encoder( num_nodes=num_nodes, k_num=k_num,l=l, length=length,
                               in_dim=in_dim, residual_channels=residual_channels, dilation_channels=dilation_channels,
                               K=K, Kt=Kt)
        self.decoder = Decoder( num_nodes=num_nodes, k_num=k_num,l=l, length=length,
                               residual_channels=residual_channels, dilation_channels=dilation_channels, K=K, Kt=Kt)
    
        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.conv1d_1 = nn.Conv2d(in_channels=in_dim, out_channels=dilation_channels, kernel_size=1)
        self.conv1d_2 = nn.Conv2d(in_channels=in_dim, out_channels=dilation_channels, kernel_size=1)
  
        self.bn_1 = BatchNorm2d(in_dim, affine=False)

    def forward(self, input):

        x = input[:, :self.in_dim, :, :]
        Trend = input[:, self.in_dim:, :, :]
        trend1 = F.relu(self.conv1d_1(Trend)).permute(0, 3, 2, 1)
        trend2 = F.relu(self.conv1d_2(Trend)).permute(0, 3, 2, 1)
        Trend_Matrix = torch.softmax(torch.matmul(trend2, trend1.transpose(2, 3)), dim=-1)

        A = self.h + self.supports[0]
        d = 1 / (torch.sum(A, -1) + 0.0001)
        D = torch.diag_embed(d)
        A = torch.matmul(D, A)
        A = F.dropout(A, self.dropout, self.training)
        v=x
        v = self.encoder(v, Trend_Matrix, A)
        v = self.decoder(v, Trend_Matrix, A)
        x = self.conv1(v)
        return x