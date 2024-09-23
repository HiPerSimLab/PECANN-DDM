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


'''
def rank_domain(full_domain, full_dom_dis, rank, size, dims):
    x_min    = full_domain[0,0]
    x_max    = full_domain[1,0]
    y_min    = full_domain[0,1]
    y_max    = full_domain[1,1]

    ### Follow MPI Cart Topology ###
    nx, ny   = dims
    x_joints = np.linspace(x_min, x_max, nx+1)
    y_joints = np.linspace(y_min, y_max, ny+1)
    dom_dis = [full_dom_dis[0]//nx, full_dom_dis[1]//ny]

    x_rank = rank // ny
    y_rank = (ny-1) - rank % ny

    domain = np.vstack((x_joints[x_rank:2+x_rank],
                        y_joints[y_rank:2+y_rank] )).T
    return domain, dom_dis

def fetch_interior_data(domain, dom_dis, surrounding=False):
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

def fetch_bounds(domain, dom_dis):
    x_min    = domain[0,0]
    x_max    = domain[1,0]
    y_min    = domain[0,1]
    y_max    = domain[1,1]

    x_series = torch.linspace( x_min, x_max, dom_dis[0]+1 )
    y_series = torch.linspace( y_min, y_max, dom_dis[1]+1 )
    ### up ###
    xu     = x_series[:,None]
    yu     = torch.full((dom_dis[0]+1,1), y_max)
    bd_u   = torch.cat((xu, yu), dim = 1)
    ### down ###
    xd     = x_series[:,None]
    yd     = torch.full((dom_dis[0]+1,1), y_min)
    bd_d   = torch.cat((xd, yd), dim = 1)
    ### left ###
    yl   = y_series[:,None]
    xl   = torch.full((dom_dis[1]+1,1), x_min)
    bd_l  = torch.cat((xl, yl), dim = 1)
    ### right ###
    yr   = y_series[:,None]
    xr   = torch.full((dom_dis[1]+1,1), x_max)
    bd_r = torch.cat((xr, yr), dim = 1)
    return bd_u, bd_l, bd_d, bd_r # counter-clockwise

def fetch_boundary_data(domain, dom_dis, rank, size, dims):
    bd_u, bd_l, bd_d, bd_r = fetch_bounds(domain, dom_dis)

    ### Follow MPI Cart Topology ###
    nx, ny   = dims
    x_rank = rank // ny
    y_rank = (ny-1) - rank % ny

    data = []
    if y_rank == ny-1:
        data.append(bd_u)
    if x_rank == 0:
        data.append(bd_l)
    if y_rank == 0:
        data.append(bd_d)
    if x_rank == nx-1:
        data.append(bd_r)
    if data == []:
        data.append( torch.full((1, 2), float('nan')) )
    data   = torch.cat(data, dim = 0)

    x        = data[:,0][:,None]
    y        = data[:,1][:,None]
    return x, y

def fetch_interface_data(domain, dom_dis, rank, size, dims):
    bd_u, bd_l, bd_d, bd_r = fetch_bounds(domain, dom_dis)

    ### Follow MPI Cart Topology ###
    nx, ny   = dims
    x_rank = rank // ny
    y_rank = (ny-1) - rank % ny

    # counter-clockwise
    data = [bd_u, bd_l, bd_d, bd_r]
    idx  = np.arange(4) # four directions
    if y_rank == ny-1:
        idx  = idx[idx != 0] #up
    if x_rank == 0:
        idx  = idx[idx != 1] #left
    if y_rank == 0:
        idx  = idx[idx != 2] #down
    if x_rank == nx-1:
        idx  = idx[idx != 3] #right
    if data == []:
        data.append( torch.full((1, 2), float('nan')) )
    data = [data[i] for i in idx]
    data = torch.cat(data, dim = 1)

    x    = data[:, 0::2]
    y    = data[:, 1::2]
    return x, y

'''
