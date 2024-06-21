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

def fetch_interior_data(domain, batch_size, surrounding=False):
    dim      = len(domain)
    x_min    = domain[0,0]
    x_max    = domain[1,0]
    y_min    = domain[0,1]
    y_max    = domain[1,1]
    
    sobol    = torch.quasirandom.SobolEngine(dimension=dim,scramble=True)
    data     = sobol.draw(batch_size,dtype=torch.float64)
    x        = data[:,0][:,None] * (x_max - x_min) + x_min
    y        = data[:,1][:,None] * (y_max - y_min) + y_min
    return x,y

def fetch_bounds(domain, batch_size):
    dim      = len(domain)
    x_min    = domain[0,0]
    x_max    = domain[1,0]
    y_min    = domain[0,1]
    y_max    = domain[1,1]
    
    sobol   = torch.quasirandom.SobolEngine(dimension=1,scramble=True)
    sobol.draw(batch_size,dtype=torch.float64)
    x_series = sobol.draw(batch_size,dtype=torch.float64) * (x_max - x_min) + x_min
    y_series = sobol.draw(batch_size,dtype=torch.float64) * (y_max - y_min) + y_min
    ### up ###
    xu     = x_series.clone()
    yu     = torch.full(x_series.shape, y_max)
    bd_u   = torch.cat((xu, yu), dim = 1)
    ### down ###
    xd     = x_series.clone()
    yd     = torch.full(x_series.shape, y_min)
    bd_d   = torch.cat((xd, yd), dim = 1)
    ### left ###
    yl   = y_series.clone()
    xl   = torch.full(y_series.shape, x_min)
    bd_l  = torch.cat((xl, yl), dim = 1)
    ### right ###
    yr   = y_series.clone()
    xr   = torch.full(y_series.shape, x_max)
    bd_r = torch.cat((xr, yr), dim = 1)
    return bd_u, bd_l, bd_d, bd_r # counter-clockwise

def fetch_boundary_data(domain, batch_size, rank, size, dims):
    bd_u, bd_l, bd_d, bd_r = fetch_bounds(domain, batch_size)
    
    ### Follow MPI Cart Topology ###
    nx, ny   = dims
    x_rank = rank // ny
    y_rank = rank % ny

    data = [] # counter-clockwise
    if y_rank == 0:
        data.append(bd_u)
    if x_rank == 0:
        data.append(bd_l)
    if y_rank == ny-1:
        data.append(bd_d)
    if x_rank == nx-1:
        data.append(bd_r)
    if data == []:
        data.append( torch.full((1, 2), float('nan')) )
    data   = torch.cat(data, dim = 0)

    x        = data[:,0][:,None]
    y        = data[:,1][:,None]
    return x, y

def fetch_interface_data(domain, batch_size, rank, size, dims):
    bd_u, bd_l, bd_d, bd_r = fetch_bounds(domain, batch_size)
    
    ### Follow MPI Cart Topology ###
    nx, ny   = dims
    x_rank = rank // ny
    y_rank = rank % ny
    
    # counter-clockwise
    data = [bd_u, bd_l, bd_d, bd_r]
    idx  = np.arange(4) # four directions
    if y_rank == 0:
        idx  = idx[idx != 0] #up
    if x_rank == 0:
        idx  = idx[idx != 1] #left
    if y_rank == ny-1:
        idx  = idx[idx != 2] #down
    if x_rank == nx-1:
        idx  = idx[idx != 3] #right
    data = [data[i] for i in idx]
    
    if data == []:
        data.append( torch.full((1, 2), float('nan')) )
    data = torch.cat(data, dim = 1)

    x    = data[:, 0::2]
    y    = data[:, 1::2]
    return x, y

def fetch_uniform_mesh(domain, dom_dis, surrounding=False):
    x_min    = domain[0,0]
    x_max    = domain[1,0]
    y_min    = domain[0,1]
    y_max    = domain[1,1]
      
    x_series = torch.linspace( x_min, x_max, dom_dis[0]+1 )
    y_series = torch.linspace( y_min, y_max, dom_dis[1]+1 )
    if surrounding:
        x,y  = torch.meshgrid(x_series,y_series, indexing='xy')
    else:
        x,y  = torch.meshgrid(x_series[1:-1],y_series[1:-1], indexing='xy')
    
    x = x.flatten()[:,None]
    y = y.flatten()[:,None]
    return x,y

