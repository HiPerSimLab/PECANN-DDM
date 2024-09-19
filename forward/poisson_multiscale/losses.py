#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:01:33 2024

@author: qifenghu
"""
import torch
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
pi = torch.tensor(np.pi, device = device)

def u_exact(x,y, freq):
    u_e = torch.sin(freq[0] * pi * x) * torch.cos(freq[1] * pi * y) + torch.cos(freq[2] * pi * x) * torch.sin(freq[3] * pi * y)
    return u_e

def uxx_exact(x,y, freq):
    term1 = - (freq[0] * pi).pow(2) * torch.sin(freq[0] * pi * x) * torch.cos(freq[1] * pi * y)
    term2 = - (freq[2] * pi).pow(2) * torch.cos(freq[2] * pi * x) * torch.sin(freq[3] * pi * y)
    return term1 + term2

def uyy_exact(x,y, freq):
    term1 = - (freq[1] * pi).pow(2) * torch.sin(freq[0] * pi * x) * torch.cos(freq[1] * pi * y)
    term2 = - (freq[3] * pi).pow(2) * torch.cos(freq[2] * pi * x) * torch.sin(freq[3] * pi * y)
    return term1 + term2

def s_exact(x,y, freq):
    return uxx_exact(x,y, freq) + uyy_exact(x,y, freq)

def physics_loss(model,x,y, freq):
    u         = model(x,y)
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)
    u_xx      = torch.autograd.grad(u_x.sum(),x,create_graph=True,retain_graph=True)[0]
    u_yy      = torch.autograd.grad(u_y.sum(),y,create_graph=True,retain_graph=True)[0]
    s         = s_exact(x,y, freq)
    loss      =  (u_xx + u_yy - s).pow(2) # poisson equation
    return torch.mean(loss).reshape(1, 1)

def boundary_loss(model,x,y, freq):
    if torch.isnan(x).any():
        return torch.zeros(1,1).to(device)
    else:
        u       = model(x,y)
        u_bc    = u_exact(x,y, freq)
        loss    = (u - u_bc).pow(2)
        return torch.mean(loss).reshape(1, 1)

def avg_if_loss(Alpha, model, x, y, u_adj, dudn_adj, dudt_adj):
    n, num_interface = x.shape
    avg_loss = torch.zeros(num_interface, 1, device=device)

    for i in range(num_interface):
        x_col = x[:, i].reshape(-1, 1)
        y_col = y[:, i].reshape(-1, 1)
        u_col = model(x_col, y_col)

        all_x_same = torch.all(x_col.eq(x_col[0]))
        if all_x_same: #column has the same x
            dudn_col,dudt_col = torch.autograd.grad(u_col.sum(), (x_col,y_col), create_graph=True)
        else:
            dudt_col,dudn_col = torch.autograd.grad(u_col.sum(), (x_col,y_col), create_graph=True)

        G    = u_col - u_adj[:, i].reshape(-1, 1)  # interface dirichlete residual
        F    = dudn_col - dudn_adj[:, i].reshape(-1, 1) # interface flux residual
        T    = dudt_col - dudt_adj[:, i].reshape(-1, 1) # interface tangential derivative

        avg_loss[i] =  (Alpha[0] * G).pow(2).mean() + (Alpha[1] * F).pow(2).mean() + (Alpha[2] * T).pow(2).mean()
    return torch.mean(avg_loss).reshape(-1,1)

