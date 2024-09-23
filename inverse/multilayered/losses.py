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

def u_exact(x,y):
    u_e = torch.zeros_like(x)

    idx = x<0.5
    u_e[idx] = 88 * x[idx]**2 * (1-x[idx]) * y[idx]
    idx = x>=0.5
    u_e[idx] = (50*x[idx]-3) * (1-x[idx]) * y[idx]
    return u_e

def k_exact(x,y):
    k_e = torch.zeros_like(x)

    k_e[x<0.5]  = 3/22
    k_e[x>=0.5] = 3/3
    return k_e

def uxx_exact(x,y):
    uxx_e = torch.zeros_like(x)

    idx = x<0.5
    uxx_e[idx] = 88 * y[idx] * (2-6*x[idx])
    idx = x>=0.5
    uxx_e[idx] = -100 * y[idx]
    return uxx_e

def uyy_exact(x,y):
    return torch.zeros_like(x)

def s_exact(x,y):
    return k_exact(x,y) * uxx_exact(x,y) + k_exact(x,y)* uyy_exact(x,y)

def physics_loss(model,x,y):
    u         = model(x,y)
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)
    flux_x    = model.k * u_x
    flux_y    = model.k * u_y
    u_xx      = torch.autograd.grad(flux_x.sum(),x,create_graph=True,retain_graph=True)[0]
    u_yy      = torch.autograd.grad(flux_y.sum(),y,create_graph=True,retain_graph=True)[0]
    s         = s_exact(x,y)
    loss      =  (u_xx + u_yy - s).pow(2) 
    return torch.mean(loss).reshape(1, 1)

def boundary_loss(model,x,y):
    if torch.isnan(x).any():
        return torch.zeros(1,1).to(device)
    else:
        u       = model(x,y)
        u_bc    = u_exact(x,y)
        loss    = (u - u_bc).pow(2)
        return torch.mean(loss).reshape(1, 1)

def avg_if_loss(Alpha, model, x, y, u_adj, dudn_adj, dudt_adj):
    n, num_interface = x.shape
    avg_loss = torch.zeros(num_interface, 1, device=device)

    for i in range(num_interface):
        x_col = x[:, i].reshape(-1, 1)
        y_col = y[:, i].reshape(-1, 1)
        u_col = model(x_col, y_col)
        u_x_col,u_y_col   = torch.autograd.grad(u_col.sum(),(x_col,y_col),create_graph=True,retain_graph=True)

        all_x_same = torch.all(x_col.eq(x_col[0]))
        if all_x_same: #column has the same x
            dudn_col = model.k * u_x_col
            dudt_col = u_y_col
        else:
            dudn_col = model.k * u_y_col
            dudt_col = u_x_col

        G    = u_col - u_adj[:, i].reshape(-1, 1)  # interface dirichlete residual
        F    = dudn_col - dudn_adj[:, i].reshape(-1, 1) # interface flux residual
        T    = dudt_col - dudt_adj[:, i].reshape(-1, 1) # interface tangential derivative

        avg_loss[i] =  (Alpha[0] * G).pow(2).mean() + (Alpha[1] * F).pow(2).mean() + (Alpha[2] * T).pow(2).mean()
    return torch.mean(avg_loss).reshape(-1,1)

