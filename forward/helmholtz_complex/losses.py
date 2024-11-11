#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:01:33 2024

@author: qifenghu
"""
import torch
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cpu")
pi = torch.tensor(np.pi, device = device)

def u_exact(x, y, k, theta):
    k1 = k * torch.cos(theta)
    k2 = k * torch.sin(theta)
    
    u_re = torch.cos( k1 * x + k2 * y)
    u_im = torch.sin( k1 * x + k2 * y)
    return torch.cat((u_re, u_im), dim=1)


def u_x_exact(x, y, k, theta):
    k1 = k * torch.cos(theta)
    k2 = k * torch.sin(theta)
    
    u_x_re = - k1 * torch.sin( k1 * x + k2 * y)
    u_x_im = k1 * torch.cos( k1 * x + k2 * y)
    return torch.cat((u_x_re, u_x_im), dim=1)


def u_y_exact(x, y, k, theta):
    k1 = k * torch.cos(theta)
    k2 = k * torch.sin(theta)
    
    u_y_re = - k2 * torch.sin( k1 * x + k2 * y)
    u_y_im = k2 * torch.cos( k1 * x + k2 * y)
    return torch.cat((u_y_re, u_y_im), dim=1)


def physics_loss(model,x,y,k):
    u          = model(x,y)
    u_re, u_im = u[:, 0:1], u[:, 1:2]
    
    u_re_x, u_re_y = torch.autograd.grad(u_re.sum(),(x,y),create_graph=True,retain_graph=True)
    u_im_x, u_im_y = torch.autograd.grad(u_im.sum(),(x,y),create_graph=True,retain_graph=True)
    u_re_xx = torch.autograd.grad(u_re_x.sum(),x,create_graph=True,retain_graph=True)[0]
    u_im_xx = torch.autograd.grad(u_im_x.sum(),x,create_graph=True,retain_graph=True)[0]
    u_re_yy = torch.autograd.grad(u_re_y.sum(),y,create_graph=True,retain_graph=True)[0]
    u_im_yy = torch.autograd.grad(u_im_y.sum(),y,create_graph=True,retain_graph=True)[0]
    
    loss      = (u_re_xx + u_re_yy + k**2 * u_re - 0).pow(2) + (u_im_xx + u_im_yy + k**2 * u_im - 0).pow(2) # square magnitude of helmholtz residual for complex values
    return torch.mean(loss).reshape(1, 1)


def boundary_loss(model,x,y, k, theta):
    u          = model(x, y)
    u_re, u_im = u[:, 0:1], u[:, 1:2]
    
    u_x_re, u_y_re = torch.autograd.grad(u_re.sum(),(x,y),create_graph=True,retain_graph=True)
    u_x_im, u_y_im = torch.autograd.grad(u_im.sum(),(x,y),create_graph=True,retain_graph=True)
    
    u_ex   = u_exact(x, y, k, theta)
    u_x_ex = u_x_exact(x, y, k, theta)
    u_y_ex = u_y_exact(x, y, k, theta)
    u_ex_re, u_ex_im = u_ex[:, 0:1], u_ex[:, 1:2]
    u_x_ex_re, u_x_ex_im = u_x_ex[:, 0:1], u_x_ex[:, 1:2]
    u_y_ex_re, u_y_ex_im = u_y_ex[:, 0:1], u_y_ex[:, 1:2]
    
    mask1    = y==0. # out normal direction: -y
    avg_loss1= torch.zeros((1,1), device=device)
    if mask1.sum() != 0:
        loss1_re = ( (-k*u_im[mask1]   -u_y_re[mask1]) - 
                 (-k*u_ex_im[mask1]-u_y_ex_re[mask1]) ).pow(2)
        loss1_im = ( ( k*u_re[mask1]   -u_y_im[mask1]) - 
                 ( k*u_ex_re[mask1]-u_y_ex_im[mask1]) ).pow(2)
        avg_loss1= torch.mean( loss1_re + loss1_im ).reshape(1, 1)
    
    mask2    = x==1. # out normal direction: x
    avg_loss2= torch.zeros((1,1), device=device)
    if mask2.sum() != 0:
        loss2_re = ( (-k*u_im[mask2]   +u_x_re[mask2]) - 
                 (-k*u_ex_im[mask2]+u_x_ex_re[mask2]) ).pow(2)
        loss2_im = ( ( k*u_re[mask2]   +u_x_im[mask2]) - 
                 ( k*u_ex_re[mask2]+u_x_ex_im[mask2]) ).pow(2)
        avg_loss2= torch.mean( loss2_re + loss2_im ).reshape(1, 1)

    mask3    = y==1. # out normal direction: y
    avg_loss3= torch.zeros((1,1), device=device)
    if mask3.sum() != 0:
        loss3_re = ( (-k*u_im[mask3]   +u_y_re[mask3]) - 
                 (-k*u_ex_im[mask3]+u_y_ex_re[mask3]) ).pow(2)
        loss3_im = ( ( k*u_re[mask3]   +u_y_im[mask3]) - 
                 ( k*u_ex_re[mask3]+u_y_ex_im[mask3]) ).pow(2)
        avg_loss3= torch.mean( loss3_re + loss3_im ).reshape(1, 1)
    
    mask4    = x==0. # out normal direction: -x
    avg_loss4= torch.zeros((1,1), device=device)
    if mask4.sum() != 0:
        loss4_re = ( (-k*u_im[mask4]   -u_x_re[mask4]) - 
                 (-k*u_ex_im[mask4]-u_x_ex_re[mask4]) ).pow(2)
        loss4_im = ( ( k*u_re[mask4]   -u_x_im[mask4]) - 
                 ( k*u_ex_re[mask4]-u_x_ex_im[mask4]) ).pow(2)
        avg_loss4= torch.mean( loss4_re + loss4_im ).reshape(1, 1)
    
    avg_loss = torch.mean( avg_loss1 + avg_loss2 + avg_loss3 + avg_loss4 ).reshape(1, 1) 
    return avg_loss


def interface_loss(q, model, x,y, u_adj, dudn_adj, dudt_adj):
    if torch.isnan(x).any():
        return torch.zeros_like(x)
    else:
        n, num_interface = x.shape
        avg_loss = torch.zeros(num_interface, 1, device=device)

        for i in range(num_interface):
            x_side = x[:, i:i+1]
            y_side = y[:, i:i+1]
            u_side = model(x_side, y_side)
            u_side_re, u_side_im = u_side[:,0:1], u_side[:,1:2]

            all_x_same = torch.all(x_side.eq(x_side[0]))
            if all_x_same: #column has the same x
                dudn_side_re, dudt_side_re = torch.autograd.grad(u_side_re.sum(),(x_side, y_side), create_graph=True,retain_graph=True)
                dudn_side_im, dudt_side_im = torch.autograd.grad(u_side_im.sum(),(x_side, y_side), create_graph=True,retain_graph=True)
            else:
                dudt_side_re, dudn_side_re = torch.autograd.grad(u_side_re.sum(), (x_side,y_side), create_graph=True,retain_graph=True)
                dudt_side_im, dudn_side_im = torch.autograd.grad(u_side_im.sum(), (x_side,y_side), create_graph=True,retain_graph=True)

            dudn_side    = torch.cat((dudn_side_re, dudn_side_im), dim=1)
            dudt_side    = torch.cat((dudt_side_re, dudt_side_im), dim=1)

            G = u_side - u_adj[i] # interface dirichlet residual (complex values)
            F = dudn_side - dudn_adj[i] # interface neumann residual
            T = dudt_side - dudt_adj[i] # interface tangential continuity

            avg_loss[i] = ( q[0].pow(2) * G.pow(2).sum(dim=1) 
                          + q[1].pow(2) * F.pow(2).sum(dim=1) 
                          + q[2].pow(2) * T.pow(2).sum(dim=1) ).mean()
            
        return torch.mean(avg_loss).reshape(-1,1)


