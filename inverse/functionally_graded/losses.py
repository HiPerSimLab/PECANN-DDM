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
    # for interface 2d tensor data
    if torch.isnan(x).any():
        return torch.zeros_like(x)
    elif x.shape[1] != 1:
        x = x.T.reshape(-1,1)
        y = x.T.reshape(-1,1)
    U         = model(x,y) # U contains T, k
    u         = U[:,0][:,None]
    k         = U[:,1][:,None]
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)
    ku_xx     = torch.autograd.grad((k*u_x).sum(),x,create_graph=True,retain_graph=True)[0]
    ku_yy     = torch.autograd.grad((k*u_y).sum(),y,create_graph=True,retain_graph=True)[0]
    s         = s_exact(x,y)
    loss      = (ku_xx + ku_yy - s).pow(2)
    return loss

def boundary_loss(model,x,y):
    if torch.isnan(x).any():
        return torch.zeros_like(x)
    else:
        U       = model(x,y)
        u       = U[:,0][:,None]
        k       = U[:,1][:,None]

        u_bc    = u_exact(x,y)
        k_bc    = k_exact(x,y)
        
        e1      = (u - u_bc).pow(2)
        e2      = (k - k_bc).pow(2)
        return e2

def avg_if_loss(Robin, model, x, y, u_adj, k_adj, du_adj):
    if torch.isnan(x).any():
        return torch.zeros_like(x)
    else:
        n, num_interface = x.shape
        avg_loss = torch.zeros(num_interface, 1, device=device)
        
        for i in range(num_interface):
            x_col = x[:, i].reshape(-1, 1)
            y_col = y[:, i].reshape(-1, 1)
            U_col = model(x_col, y_col)
            u_col = U_col[:,0][:,None]
            k_col = U_col[:,1][:,None]

            all_x_same = torch.all(x_col.eq(x_col[0]))
            if all_x_same: #column has the same x
                du_col = k_col * torch.autograd.grad(u_col.sum(), x_col, create_graph=True)[0]
            else:
                du_col = k_col * torch.autograd.grad(u_col.sum(), y_col, create_graph=True)[0]

            N    = du_col - du_adj[:, i].reshape(-1, 1) # interface neumann residual
            D    = u_col - u_adj[:, i].reshape(-1, 1)  # interface dirichlet residual - for temperature
            
            # Robin[0]: Dirichlet; [1]: Neumann
            avg_loss[i] = ( Robin[0].pow(2) * D.pow(2) ).mean() + (Robin[1] * N).pow(2).mean() 
        return torch.mean(avg_loss).reshape(-1,1)

def measurement_loss(model,x,y):
    if torch.isnan(x).any():
        return torch.zeros_like(x)
    else:
        U       = model(x,y)
        u       = U[:,0][:,None]
        k       = U[:,1][:,None]

        u_e     = u_exact(x,y)
        k_e     = k_exact(x,y)

        e1      = (u - u_e).pow(2)
        e2      = (k - k_e).pow(2)
        return e1, e2
