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

def u_exact(x,y):
    u = torch.zeros_like(x).detach()
    for m in range(1, 50+1):
        for n in range(1, 50+1):
            lambda_mn = (m**2+n**2) * np.pi**2
            integral_x = - (np.cos(3*m*np.pi/4)-np.cos(m*np.pi/4)) / (m*np.pi)
            integral_y = - (np.cos(3*n*np.pi/4)-np.cos(n*np.pi/4)) / (n*np.pi)
            Amn = -4/lambda_mn * integral_x * integral_y
            u += Amn * torch.sin(m * np.pi * x) * torch.sin(n * np.pi * y)
    return u

def s_exact(x,y):
    s       = torch.zeros_like(x).detach().to(device)
    mask    = (x >= 0.25) & (x <= 0.75) & (y >= 0.25) & (y <= 0.75)
    s[mask] = 1
    return s

def physics_loss(model,x,y):
    u         = model(x,y)
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)
    u_xx      = torch.autograd.grad(u_x.sum(),x,create_graph=True,retain_graph=True)[0]
    u_yy      = torch.autograd.grad(u_y.sum(),y,create_graph=True,retain_graph=True)[0]
    s         = s_exact(x,y)
    loss      =  (u_xx + u_yy - s).pow(2) # poisson equation
    return loss


def boundary_loss(model,x,y):
    if torch.isnan(x).any():
        return torch.zeros_like(x).to(device)
    else:
        u       = model(x,y)
        u_bc    = 0.
        e       = u - u_bc
        return e.pow(2)

def avg_if_loss(Robin, model, x, y, u_adj, du_adj):
    u    = model(x, y)
    du   = torch.zeros_like(x).to(device)
    
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)

    mask = y == 0.25 # Bottom edge (x, 0.25)
    du[mask] = u_y[mask]
    
    mask = x == 0.75 # Right edge (0.75, y)
    du[mask] = u_x[mask]
    
    mask = y == 0.75 # Top edge (x, 0.75)
    du[mask] = u_y[mask]
    
    mask = x == 0.25 # Left edge (0.25, y)
    du[mask] = u_x[mask]

    D    = u - u_adj  # interface dirichlet residual
    N    = du - du_adj # interface neumann residual
    
    avg_loss =  (Robin[0] * D).pow(2).mean() + (Robin[1] * N).pow(2).mean()
    return avg_loss.reshape(-1,1)
