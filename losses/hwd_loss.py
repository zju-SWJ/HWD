"""Custom losses."""
from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable


class CriterionHWD(nn.Module):
    def __init__(self, norm=True, L='MSE', process='E', parameter=0.1, DivPixel='None', shift=0.0, activation=False):
        super(CriterionHWD, self).__init__()
        self.L = L
        self.process = process
        self.parameter = parameter
        self.norm = norm
        self.DivPixel = DivPixel
        self.shift = shift
        self.activation = activation

    def get_weight(self, preds_S, preds_T, labels):
        n, c, _ = preds_S.shape
        device = preds_S.device
        count = torch.ones([n, c], dtype = torch.float).to(device) # count class-related pixel numbers
        for i in range(n):
            for j in range(c):
                count[i, j] += torch.sum(labels[i] == j)
        if self.L == 'MSE':
            mse = nn.MSELoss(reduction='none')
            dis = torch.sum(mse(preds_S, preds_T), dim=-1)
        elif self.L == 'L1':
            l1 = nn.L1Loss(reduction='none')
            dis = torch.sum(l1(preds_S, preds_T), dim=-1)
        elif self.L == 'SMOOTHL1':
            smoothl1 = nn.SmoothL1Loss(reduction='none')
            dis = torch.sum(smoothl1(preds_S, preds_T), dim=-1)
        elif self.L == 'KL':
            assert self.activation == True, 'Activation should be True when KL is used.'
            dis = torch.sum(F.kl_div(preds_S.log(), preds_T, reduction='none'), dim=-1)
        if self.DivPixel == 'N':
            weight = dis / count
        elif self.DivPixel == 'SqrtN':
            weight = dis / torch.sqrt(count)
        elif self.DivPixel == 'N2':
            weight = dis / torch.pow(count, 2)
        else:
            weight = dis
        if self.norm == True: # normalization
            weight_mean = weight.mean(dim=1, keepdim=True).repeat(1, c)
            weight_std = weight.std(dim=1, keepdim=True).repeat(1, c)
            weight = (weight - weight_mean) / weight_std + self.shift
        if self.process == 'E': # smoothing, softmax
            final_weight = (1 - self.parameter ** weight) / (1 - self.parameter)
            final_weight = F.softmax(final_weight, dim=-1) * c
        elif self.process == 'T': # skip smoothing, high temperature softmax
            final_weight = F.softmax(weight / self.parameter, dim=-1) * c
        return final_weight
            
    def forward(self, preds_S, preds_T, labels):
        n, c, h, w = preds_S.shape
        labels = labels.unsqueeze(1).float().clone()
        labels = F.interpolate(labels, (h, w), mode='nearest')
        labels = labels.squeeze(1).long()
        preds_S = preds_S.reshape((n, c, -1))
        preds_T = preds_T.reshape((n, c, -1))
        
        if self.activation == False:
            with torch.no_grad():
                final_weight = self.get_weight(preds_S.clone(), preds_T, labels)
            preds_S = F.softmax(preds_S / 4.0, dim=-1)
            preds_T = F.softmax(preds_T / 4.0, dim=-1)
        else:
            preds_S = F.softmax(preds_S / 4.0, dim=-1)
            preds_T = F.softmax(preds_T / 4.0, dim=-1)
            with torch.no_grad():
                final_weight = self.get_weight(preds_S.clone(), preds_T, labels)

        loss = F.kl_div(preds_S.log(), preds_T.detach(), reduction='none') * (4.0**2)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mul(final_weight, loss)
        loss = torch.sum(loss)
        loss /= n * c
        return loss


class CriterionHWD_SPATIAL(nn.Module): # to be studied, do NOT use
    def __init__(self, norm=True, L='MSE', process='E', parameter=0.1, shift=0.0, activation=False):
        super(CriterionHWD_SPATIAL, self).__init__()
        self.L = L
        self.process = process
        self.parameter = parameter
        self.norm = norm
        self.shift = shift
        self.activation = activation

    def get_weight(self, preds_S, preds_T, hw):
        n, c, _ = preds_S.shape
        if self.L == 'MSE':
            mse = nn.MSELoss(reduction='none')
            dis = torch.sum(mse(preds_S, preds_T), dim=1)
        elif self.L == 'L1':
            l1 = nn.L1Loss(reduction='none')
            dis = torch.sum(l1(preds_S, preds_T), dim=1)
        elif self.L == 'SMOOTHL1':
            smoothl1 = nn.SmoothL1Loss(reduction='none')
            dis = torch.sum(smoothl1(preds_S, preds_T), dim=1)
        weight = dis
        if self.norm == True:
            weight_mean = weight.mean(dim=1, keepdim=True).repeat(1, hw)
            weight_std = weight.std(dim=1, keepdim=True).repeat(1, hw)
            weight = (weight - weight_mean) / weight_std + self.shift
        if self.process == 'E':
            final_weight = (1 - self.parameter ** weight) / (1 - self.parameter)
            final_weight = F.softmax(final_weight, dim=-1).unsqueeze(1).repeat(1, c, 1)
        elif self.process == 'T':
            final_weight = F.softmax(weight / self.parameter, dim=-1).unsqueeze(1).repeat(1, c, 1)
        return final_weight
            
    def forward(self, preds_S, preds_T, labels):
        n, c, h, w = preds_S.shape
        preds_S = preds_S.reshape((n, c, -1))
        preds_T = preds_T.reshape((n, c, -1))
        
        if self.activation == False:
            with torch.no_grad():
                final_weight = self.get_weight(preds_S.clone(), preds_T, h*w)
            preds_S = F.softmax(preds_S / 4.0, dim=-1)
            preds_T = F.softmax(preds_T / 4.0, dim=-1)
        else:
            preds_S = F.softmax(preds_S / 4.0, dim=-1)
            preds_T = F.softmax(preds_T / 4.0, dim=-1)
            with torch.no_grad():
                final_weight = self.get_weight(preds_S.clone(), preds_T, h*w)

        preds_T = final_weight * preds_T
        preds_T = preds_T / preds_T.sum(dim=-1, keepdim=True).repeat(1, 1, h*w)
        loss = F.kl_div(preds_S.log(), preds_T.detach(), reduction='sum') * (4.0**2)
        loss /= n * c
        return loss
