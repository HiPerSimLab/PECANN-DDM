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

def u_exact(x,y, omega):
    u_e = torch.sin(omega[0] * pi * x) * torch.cos(omega[1] * pi * y) + torch.cos(omega[2] * pi * x) * torch.sin(omega[3] * pi * y)
    return u_e

def uxx_exact(x,y, omega):
    term1 = - (omega[0] * pi).pow(2) * torch.sin(omega[0] * pi * x) * torch.cos(omega[1] * pi * y)
    term2 = - (omega[2] * pi).pow(2) * torch.cos(omega[2] * pi * x) * torch.sin(omega[3] * pi * y)
    return term1 + term2

def uyy_exact(x,y, omega):
    term1 = - (omega[1] * pi).pow(2) * torch.sin(omega[0] * pi * x) * torch.cos(omega[1] * pi * y)
    term2 = - (omega[3] * pi).pow(2) * torch.cos(omega[2] * pi * x) * torch.sin(omega[3] * pi * y)
    return term1 + term2

def s_exact(x,y, omega):
    return uxx_exact(x,y, omega) + uyy_exact(x,y, omega) + 1. * u_exact(x,y, omega)

def physics_loss(model,x,y, omega):
    # for interface 2d tensor data
    if torch.isnan(x).any():
        return torch.zeros_like(x)
    elif x.shape[1] != 1:
        x = x.T.reshape(-1,1)
        y = x.T.reshape(-1,1)
    u         = model(x,y)
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)
    u_xx      = torch.autograd.grad(u_x.sum(),x,create_graph=True,retain_graph=True)[0]
    u_yy      = torch.autograd.grad(u_y.sum(),y,create_graph=True,retain_graph=True)[0]
    s         = s_exact(x,y, omega)
    loss      =  (u_xx + u_yy + 1.*u - s).pow(2) # helmholtz equation
    return loss

def boundary_loss(model,x,y, omega):
    if torch.isnan(x).any():
        return torch.zeros_like(x)
    else:
        u       = model(x,y)
        u_bc    = u_exact(x,y, omega)
        e       = u - u_bc
        return e.pow(2)

def avg_if_loss(Alpha, model, x, y, u_adj, du_adj):
    if torch.isnan(x).any():
        return torch.zeros_like(x)
    else:
        n, num_interface = x.shape
        avg_loss = torch.zeros(num_interface, 1, device=device)
        
        for i in range(num_interface):
            x_col = x[:, i].reshape(-1, 1)
            y_col = y[:, i].reshape(-1, 1)
            u_col = model(x_col, y_col)

            all_x_same = torch.all(x_col.eq(x_col[0]))
            if all_x_same: #column has the same x
                du_col = torch.autograd.grad(u_col.sum(), x_col, create_graph=True)[0]
            else:
                du_col = torch.autograd.grad(u_col.sum(), y_col, create_graph=True)[0]

            N    = du_col - du_adj[:, i].reshape(-1, 1) # interface neumann residual
            D    = u_col - u_adj[:, i].reshape(-1, 1)  # interface dirichlete residual

            avg_loss[i] =  (Alpha[0] * D).pow(2).mean() + (Alpha[1] * N).pow(2).mean()
        return torch.mean(avg_loss).reshape(-1,1)

