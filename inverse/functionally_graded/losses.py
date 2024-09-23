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
    u_e = 20. * torch.exp(-0.1 * y)
    return u_e

def k_exact(x,y):
    return 20. + torch.exp(0.1 * y) * torch.sin(0.5 * x)

def s_exact(x,y):
    return 4. * torch.exp(-0.1 * y)

def physics_loss(model,x,y):
    U         = model(x,y)
    u         = U[:,0][:,None]
    k         = U[:,1][:,None]
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)
    flux_x    = k * u_x
    flux_y    = k * u_y
    u_xx      = torch.autograd.grad(flux_x.sum(),x,create_graph=True,retain_graph=True)[0]
    u_yy      = torch.autograd.grad(flux_y.sum(),y,create_graph=True,retain_graph=True)[0]
    s         = s_exact(x,y)
    loss      =  (u_xx + u_yy - s).pow(2) 
    return torch.mean(loss).reshape(1, 1)

def boundary_loss(model,x,y):
    if torch.isnan(x).any():
        return torch.zeros(1,1).to(device), torch.zeros(1,1).to(device)
    else:
        U       = model(x,y)
        u       = U[:,0][:,None]
        k       = U[:,1][:,None]

        u_bc    = u_exact(x,y)
        k_bc    = k_exact(x,y)

        e1      = (u - u_bc).pow(2)
        e2      = (k - k_bc).pow(2)
        return torch.mean(e1).reshape(1, 1), torch.mean(e2).reshape(1, 1)

def avg_if_loss(Alpha, model, x, y, u_adj, k_adj, dudn_adj, dudt_adj):
    n, num_interface = x.shape
    avg_loss = torch.zeros(num_interface, 1, device=device)

    for i in range(num_interface):
        x_col = x[:, i].reshape(-1, 1)
        y_col = y[:, i].reshape(-1, 1)
        U_col = model(x_col, y_col)
        u_col = U_col[:,0][:,None]
        k_col = U_col[:,1][:,None]
        u_x_col,u_y_col   = torch.autograd.grad(u_col.sum(),(x_col,y_col),create_graph=True,retain_graph=True)

        all_x_same = torch.all(x_col.eq(x_col[0]))
        if all_x_same: #column has the same x
            dudn_col = k_col * u_x_col
            dudt_col = u_y_col
        else:
            dudn_col = k_col * u_y_col
            dudt_col = u_x_col

        G    = u_col - u_adj[:, i].reshape(-1, 1)  # interface dirichlete residual
        F    = dudn_col - dudn_adj[:, i].reshape(-1, 1) # interface flux residual
        T    = dudt_col - dudt_adj[:, i].reshape(-1, 1) # interface tangential derivative
        Gk   = k_col - k_adj[:, i].reshape(-1, 1)  # interface dirichlete residual for k
        avg_loss[i] =  (Alpha[0] * G).pow(2).mean() + (Alpha[1] * F).pow(2).mean() + (Alpha[2] * T).pow(2).mean() + (Alpha[3] * Gk).pow(2).mean()
    return torch.mean(avg_loss).reshape(-1,1)

