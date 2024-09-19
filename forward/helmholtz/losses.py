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


def s_exact(x,y, L):
    sd = 0.8 / 2**L

    s = (1/(2*pi*sd**2))*torch.exp(-0.5*(((x-0.5)/sd)**2 + ((y-0.5)/sd)**2))
    return s

def physics_loss(model,x,y, L):
    u         = model(x,y)
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)
    u_xx      = torch.autograd.grad(u_x.sum(),x,create_graph=True,retain_graph=True)[0]
    u_yy      = torch.autograd.grad(u_y.sum(),y,create_graph=True,retain_graph=True)[0]
    s         = s_exact(x,y, L)
    k         = 2**L *pi / 1.6
    loss      =  (u_xx + u_yy + k**2 * u - s).pow(2) # helmholtz equation
    return torch.mean(loss).reshape(1, 1)

def boundary_loss(model,x,y, L):
    if torch.isnan(x).any():
        return torch.zeros(1,1).to(device)
    else:
        u       = model(x,y)
        u_bc    = 0. #u_exact(x,y, L)
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

