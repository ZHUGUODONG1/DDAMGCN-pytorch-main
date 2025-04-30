 # -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()


class DGCN(nn.Module):

     def __init__(self, c_in, c_out, K, Kt):
         super(DGCN, self).__init__()
         c_in_new = (K) * c_in
         self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                             stride=(1, 1), bias=True)
         self.K = K

     def forward(self, x, adj):
         nSample, feat_in, nNode, length = x.shape
         Ls = []
         L1 = adj
         L0 = torch.eye(nNode).cuda()
         Ls.append(L0)
         Ls.append(L1)
         for k in range(2, self.K):
             L2 = 2 * torch.matmul(adj, L1) - L0
             L0, L1 = L1, L2
             Ls.append(L2)

         Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
         # print(Lap)
         Lap = Lap.transpose(-1, -2)
         x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
         x = x.view(nSample, -1, nNode, length)
         out = self.conv1(x)
         return out


class Trend_GCN(nn.Module):
    def __init__(self, in_channels, out_channels,K):
        super(Trend_GCN, self).__init__()
        c_in_new = (K-1) * in_channels
        self.K=K
        self.bn = nn.BatchNorm2d(out_channels)
        self.graph_conv_out = nn.Conv2d(in_channels=c_in_new, out_channels=out_channels, kernel_size=1)

    def forward(self, x,Trend_Matrix):
        result=x
        Ls = []
        for k in range(1, self.K):
            result = torch.einsum('adcb,abce->adbe', result, Trend_Matrix).permute(0,1,3,2)
            Ls.append(result)
        result = torch.cat(Ls, dim=1)
        out=self.graph_conv_out(result)
        out = self.bn(out)
        return out

class TCN(nn.Module):
    def __init__(self, c_in, c_out, Kt):
        super(TCN, self).__init__()
        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.conv2_5 = Conv2d(c_in, c_out, kernel_size=(1, 5), padding=(0, 2),
                              stride=(1, 1), bias=True)
        self.conv3_7 = Conv2d(c_in, c_out, kernel_size=(1, 7), padding=(0, 3),
                              stride=(1, 1), bias=True)
        self.conv4_9 = Conv2d(c_in, c_out, kernel_size=(1, 9), padding=(0, 4),
                              stride=(1, 1), bias=True)
        self.conv_out = Conv2d(4 * c_out, c_out, kernel_size=(1, 1), padding=(0, 0),
                               stride=(1, 1), bias=True)
    def forward(self, x):
        x1 = self.conv1(x)
        x2_5 = self.conv2_5(x)
        x3_7 = self.conv3_7(x)
        x4_9 = self.conv4_9(x)
        x_m = torch.cat([x1, x2_5, x3_7, x4_9], dim=1)
        x_m = self.conv_out(x_m)
        return x_m

class Gate_fusion(nn.Module):
    def __init__(self,c_out):
        super(Gate_fusion, self).__init__()

        self.conv_out = Conv2d(3 * c_out, 2*c_out, kernel_size=1)
        self.c_out=c_out

    def forward(self,z1,z2,x_input1):

        z=torch.cat((z1,z2),dim=1)
        z=self.conv_out(z)
        # self.check_for_invalid_values(z1)
        # mean_z1 = z1.mean()
        # mean_z2 = z2.mean()
        # print(mean_z1,mean_z2)
        filter, gate = torch.split(z, [self.c_out, self.c_out], 1)
        out = (filter + x_input1) * torch.sigmoid(gate)
        return out


class MSTGCN(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(MSTGCN, self).__init__()
        self.conv_1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                             stride=(1, 1), bias=True)
        self.dgcn = DGCN(c_out, 2*c_out, K, 1)
        self.tgcn=Trend_GCN(c_in, c_out,K)
        self.tcn=TCN(c_in, c_out,Kt)
        self.gf=Gate_fusion(c_out)
    def forward(self, x, Trend_Matrix,supports):
        x_input1 = self.conv_1(x)
        z1=self.tgcn(x,Trend_Matrix)
        x_m = self.tcn(x)
        z2 = self.dgcn(x_m, supports)
        x=self.gf(z1,z2,x_input1)
        return x

def generalized_cross_correlation(x, y):

    X = torch.fft.fft(x, dim=-1)
    Y = torch.fft.fft(y, dim=-1)
    R = torch.fft.ifft(X * torch.conj(Y), dim=-1)
    R = torch.real(R)
    return R

class CriticalNodeDelayCalculator(nn.Module):

    def __init__(self, K=10,input_dim=32,tem_size=12):
        super(CriticalNodeDelayCalculator, self).__init__()
        self.K = K
        # self.linear_layer = nn.Linear(K, 1)
        self.linear_layer = nn.Conv2d(K, 1, kernel_size=1)
        self.time_pro = nn.Conv3d(in_channels=12, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.time_linear_layer = nn.Linear(12, 1)
        self.FC1 = nn.Conv2d(1, 32, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.FC2 = nn.Conv2d(1, 32, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.weights = nn.Parameter(torch.randn(12))
        self.scale=torch.arange(1, 13, dtype=torch.float32).view(1,12 , 1, 1, 1)
        self.conv_out = Conv2d(2*input_dim, input_dim, kernel_size=1)
    def rolling(delf, input, peak_lags):
        peak_lags = peak_lags.unsqueeze(-1).expand(-1, input.shape[1], -1,
                                                   input.shape[-1])

        time_indices = torch.arange(input.shape[1], device=input.device)
        time_indices = time_indices.view(1, -1, 1, 1).expand_as(input)

        rolled_indices = (time_indices - peak_lags) % input.shape[1]

        rolled_data = torch.gather(input, dim=1, index=rolled_indices.long())

        return rolled_data

    def check_for_invalid_values(self, tensor):
        if torch.isnan(tensor).any():
            print(f"Warning contains NaN values!")
        if torch.isinf(tensor).any():
            print(f"Warning:  contains Inf values!")
        else:
            print(f" is valid (no NaN or Inf values).")

    def forward(self, data):
        emb_data=data.permute(0, 3, 2, 1)
        input_data=emb_data[:,:,:,:1]
        length = input_data.shape[1]
        X1=self.FC1(input_data.permute(0,3,2,1)).permute(0,3,2,1)
        X2=self.FC2(input_data.permute(0,3,2,1)).permute(0,3,2,1)
        relation_matrix=torch.softmax(torch.matmul(X1, X2.transpose(-1, -2)),dim=-1)
        normalized_relation_matrix = relation_matrix/torch.sum(relation_matrix, dim=-1, keepdim=True)
        node_influence = torch.sum(normalized_relation_matrix, dim=-1)
        top_k_nodes = torch.topk(node_influence, k=self.K, dim=-1).indices.unsqueeze(-1)
        selected_values = torch.gather(input_data, dim=2, index=top_k_nodes)
        t = 0
        delay_all_time=[]
        while t < length:
            selected_x_window = selected_values[:,t:,:]
            x_window = input_data[:, t:, :]
            key_node_data=selected_x_window.unsqueeze(3)
            all_node_data=x_window.unsqueeze(2)
            R = generalized_cross_correlation(key_node_data, all_node_data)
            R=R.squeeze(-1).permute(0,2,3,1)
            R[torch.isnan(R)] = 0
            Aggregated_R=self.linear_layer(R)
            lagc = torch.arange(- (length-t)// 2 + 1, (length-t) // 2 + 1).cuda()
            peaks = torch.argmax(Aggregated_R, dim=-1)
            tau = lagc[peaks]
            rolled_data = self.rolling(emb_data[:,t:,:,:], tau)
            rolled_data = F.pad(rolled_data, (0, 0, 0, 0, t, 0))
            delay_all_time.append(rolled_data)
            t = t + 1
        a=torch.stack(delay_all_time,dim=1)
        scale_w = torch.arange(1, 13, dtype=torch.float32).view(1, 12, 1, 1, 1).cuda()
        scaled_data = (a / scale_w).sum(dim=1).squeeze(1).permute(0,3,2,1)
        D = torch.cat((scaled_data, data), dim=1)
        D=self.conv_out(D)
        return D
