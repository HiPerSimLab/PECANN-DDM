#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:58:33 2024

@author: qifenghu
"""
import torch
import numpy as np

def rank_domain(full_domain, rank, size, dims):
    x_min    = full_domain[0,0]
    x_max    = full_domain[1,0]
    y_min    = full_domain[0,1]
    y_max    = full_domain[1,1]

    ### Follow MPI Cart Topology ###
    nx, ny   = dims
    x_joints = np.linspace(x_min, x_max, nx+1)
    y_joints = np.linspace(y_min, y_max, ny+1)

    x_rank = rank // ny
    y_rank = (ny-1) - rank % ny

    domain = np.vstack((x_joints[x_rank:2+x_rank], 
                        y_joints[y_rank:2+y_rank] )).T
    return domain

def fetch_interior_data(batch_size, rank, size):
    sobol     = torch.quasirandom.SobolEngine(dimension=2,scramble=True)
    data      = sobol.draw(batch_size,dtype=torch.float64)

    portion   = 2 * torch.pi / size
    if rank < size//2:
        theta     = data[:,0][:,None] * portion + portion * (rank+1)
    else:
        theta     = data[:,0][:,None] * portion - portion * (rank-size//2)
    rho       = 1 + torch.cos(theta)*torch.sin(4*theta)
    radius    = rho*torch.sqrt(data[:,1][:,None])
    x         = 6*0.55*radius*torch.cos(theta)
    y         = 6*0.75*radius*torch.sin(theta)

    return x, y

def fetch_boundary_data(batch_size, rank, size):
    sobol     = torch.quasirandom.SobolEngine(dimension=1,scramble=True)
    data      = sobol.draw(batch_size,dtype=torch.float64)

    portion   = 2 * torch.pi / size
    if rank < size//2:
        theta     = data[:,0][:,None] * portion + portion * (rank+1)
    else:
        theta     = data[:,0][:,None] * portion - portion * (rank-size//2)
    rho       = 1 + torch.cos(theta)*torch.sin(4*theta)
    radius    = rho
    x         = 6*0.55*radius*torch.cos(theta)
    y         = 6*0.75*radius*torch.sin(theta)
    return x,y


def fetch_interface_data(batch_size, rank, size):
    portion = 2 * torch.pi / size
    ### counter-clockwise (up, left, down, right)
    if rank==0:
        theta  = torch.linspace(2,1,2) * portion
    elif rank==1:
        theta  = torch.linspace(2,3,2) * portion
    elif rank==2:
        theta  = torch.linspace(1,0,2) * portion
    else:
        theta  = torch.linspace(3,4,2) * portion
    rho   = 1 + torch.cos(theta)*torch.sin(4*theta)

    sobol   = torch.quasirandom.SobolEngine(dimension=2,scramble=True)
    data    = sobol.draw(batch_size,dtype=torch.float64)
    radius  = rho * data
    x    = 6*0.55*radius*torch.cos(theta)
    y    = 6*0.75*radius*torch.sin(theta)
    return x, y


