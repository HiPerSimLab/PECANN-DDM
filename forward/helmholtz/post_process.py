#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata

#from sample import rank_domain, fetch_interior_data


# In[2]:


# https://joseph-long.com/writing/colorbars/
def colorbar(mappable,min_val,max_val,limit):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.2)
    ticks = np.linspace(min_val, max_val, 3, endpoint=True)
    cbar = fig.colorbar(mappable, cax=cax,ticks=ticks)
    cbar.formatter.set_powerlimits((limit, limit))
    plt.sca(last_axes)
    return cbar

params = {
    'text.latex.preamble': '\\usepackage{gensymb}',
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 12, # fontsize for x and y labels
    'axes.titlesize': 12,
    'font.size'     : 12, 
    'legend.fontsize': 12, 
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': [3, 3],
    'font.family': 'serif',
}

cmap_list = ['jet','YlGnBu','coolwarm','rainbow','magma','plasma','inferno','Spectral','RdBu']


# configurations
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update(params)

cmap = matplotlib.cm.get_cmap(cmap_list[8]).reversed() #cmap_list[8] #
torch.set_default_dtype(torch.float64)


# In[88]:


def contour_prediction(test_dis, trial, dims, size, methodname):
    u_up = np.loadtxt(f'./data/{methodname}_{trial}_x_y_u_upred.dat')

    n      = u_up.shape[0] // size
    x      = u_up[:,0].reshape(size, n).T
    y      = u_up[:,1].reshape(size, n).T
    u_star = u_up[:,2].reshape(size, n).T
    u_pred = u_up[:,3].reshape(size, n).T
    u_err  = abs(u_star - u_pred)

    ### Follow MPI Cart Topology ###
    nx, ny   = dims
    x_joints = np.linspace(x.min(), x.max(), nx+1)
    y_joints = np.linspace(y.min(), y.max(), ny+1)

    varlist = ['u_pred', 'u_star', 'u_err']
    for k, var in enumerate([u_pred, u_star, u_err]):
        fig = plt.figure(figsize = (7,6))
        gs = gridspec.GridSpec(ny, nx + 1, width_ratios=[1]*nx + [0.1], wspace=0., hspace=0.4)

        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                x_sub = x[:,rank].reshape(test_dis[1]+1, test_dis[0]+1)
                y_sub = y[:,rank].reshape(test_dis[1]+1, test_dis[0]+1)
                var_sub = var[:,rank].reshape(test_dis[1]+1, test_dis[0]+1)
                if k == 2:
                    vmax = np.max(var)
                    vmin = np.min(var)
                else:
                    vmax = np.max(u_star)
                    vmin = np.min(u_star)

                ax = plt.subplot(gs[j,i])
                pimg=plt.pcolormesh(x_sub ,y_sub ,var_sub,vmin=vmin, vmax=vmax, cmap=cmap, shading='gouraud')
                if i == 0:
                    yticks = np.linspace(y_joints[-j-2], y_joints[-j-1], 2).tolist()
                    ax.set_yticks(yticks)
                if j == ny-1:
                    xticks = np.linspace(x_joints[i], x_joints[i+1], 2).tolist()
                    ax.set_xticks(xticks)

                ax.label_outer()
                ax.axis('scaled')
                ax.title.set_text(f'({j},{i})')

        # Add colorbar to the right of the subplots
        cbar_ax = fig.add_subplot(gs[:, -1])
        ticks = np.linspace(vmin, vmax, 5, endpoint=True)
        cbar = fig.colorbar(pimg, cax=cbar_ax, ticks=ticks)
        limit = np.ceil(np.log10(vmax))-1
        cbar.formatter.set_powerlimits((limit, limit))
        cbar.update_ticks()
        plt.savefig(f'pic/{methodname}_{trial}_{varlist[k]}.png', dpi=300)


# In[102]:


def plot_update(trial, dims, size, outer_iter, methodname):
    obj_s = []
    mu_s = []
    constr_s = []
    lambda_s = []
    q_s = []
    l2_s = []
    linf_s = []
    for rank in range(size):
        obj_s.append( np.loadtxt(f'data/{trial}_{rank}_object.dat') )
        mu_s.append( np.loadtxt(f'data/{trial}_{rank}_mu.dat') )
        constr_s.append( np.loadtxt(f'data/{trial}_{rank}_constr.dat') )
        lambda_s.append( np.loadtxt(f'data/{trial}_{rank}_lambda.dat') )
        q_s.append( np.loadtxt(f'data/{trial}_{rank}_q.dat') )
        l2_s.append( np.loadtxt(f'data/{trial}_{rank}_l2.dat') )
        linf_s.append( np.loadtxt(f'data/{trial}_{rank}_linf.dat') )

    linestyles = ['-', ':', '-.', '--']
    colors = ['b', 'g', 'r', 'k']  # Blue, Black, Green, Red
    markers = ['o', 's', '^', 'd']  # Circle, Square, Triangle up, Diamond
    
    # randomly picking ranks
    ncol, nrow   = dims
    rank_locations = [(0,0), (3,1), (2,2), (4,2), (2,3), (1,4)] 
    print(rank_locations)
    
    # Plot outer iterations v.s. objectives
    fig = plt.figure(figsize = (4,4))
    for i, rank_loc  in enumerate(rank_locations):
        row = rank_loc[0]
        col = rank_loc[1]
        rank = col*nrow + row
        plt.plot(obj_s[rank][:outer_iter,0], obj_s[rank][:outer_iter,1], label=f'({row},{col})', linestyle=linestyles[i//2], color=colors[i//2], 
                linewidth=2, marker=markers[i%2], markersize=5, markevery=outer_iter//5)
    plt.xlabel('Outer Iterations', fontsize=14)
    plt.ylabel(r'$\mathcal{J}$', fontsize=14)
    plt.semilogy()
    plt.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.savefig(f'pic/{methodname}_{trial}_objective.png', dpi=300)

    # Plot outer iterations v.s. constraints
    var_list = 'constr'
    var = constr_s
    constr_list = ['BC', 'PDE']
    for n, constr_name in enumerate(constr_list):
        fig = plt.figure(figsize = (4,4))
        for i, rank_loc  in enumerate(rank_locations):
            row = rank_loc[0]
            col = rank_loc[1]
            rank = col*nrow + row
            if var[rank][0,n+1] != 0.0:
                plt.plot(var[rank][:outer_iter,0], var[rank][:outer_iter,n+1], label=f'({row},{col})', linestyle=linestyles[i//2], color=colors[i//2], 
                        linewidth=2, marker=markers[i%2], markersize=5, markevery=outer_iter//5)
        plt.xlabel('Outer Iterations', fontsize=14)
        if constr_name == 'BC':
            plt.ylabel(r'$\mathcal{C}_{B}$', fontsize=14)
        else:
            plt.ylabel(r'$\mathcal{C}_{P}$', fontsize=14)
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{var_list}_{constr_name}.png', dpi=300)

    # Plot outer iterations v.s. Lagrange multipliers
    var_list = 'lambda'
    var = lambda_s
    constr_list = ['BC', 'PDE']
    for n, constr_name in enumerate(constr_list):
        fig = plt.figure(figsize = (4,4))
        for i, rank_loc in enumerate(rank_locations):
            row = rank_loc[0]
            col = rank_loc[1]
            rank = col*nrow + row
            if constr_s[rank][-1,n+1] != 0.0:
                plt.plot(var[rank][:outer_iter,0], var[rank][:outer_iter,n+1], label=f'({row},{col})', linestyle=linestyles[i//2], color=colors[i//2], 
                         linewidth=2, marker=markers[i%2], markersize=5, markevery=outer_iter//5)
        plt.xlabel('Outer Iterations', fontsize=14)
        if constr_name == 'BC':
            plt.ylabel(r'$\lambda_{B}$', fontsize=14)
        else:
            plt.ylabel(r'$\lambda_{P}$', fontsize=14)
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{var_list}_{constr_name}.png', dpi=300)

    # Plot outer iterations v.s. penalty parameters
    var_list = 'mu'
    var = mu_s
    constr_list = ['BC', 'PDE']
    for n, constr_name in enumerate(constr_list):
        fig = plt.figure(figsize = (4,4))
        for i, rank_loc in enumerate(rank_locations):
            row = rank_loc[0]
            col = rank_loc[1]
            rank = col*nrow + row
            if constr_s[rank][-1,n+1] != 0.0:
                plt.plot(var[rank][:outer_iter,0], var[rank][:outer_iter,n+1], label=f'({row},{col})', linestyle=linestyles[i//2], color=colors[i//2], 
                         linewidth=2, marker=markers[i%2], markersize=5, markevery=outer_iter//5)

        plt.xlabel('Outer Iterations', fontsize=14)
        if constr_name == 'BC':
            plt.ylabel(r'$\mu_{B}$', fontsize=14)
        else:
            plt.ylabel(r'$\mu_{P}$', fontsize=14)
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{var_list}_{constr_name}.png', dpi=300)

    # Plot outer iterations v.s. interface parameters
    var = q_s
    inter_list = ['alpha', 'beta', 'gamma']
    for n, inter_name in enumerate(inter_list):
        fig = plt.figure(figsize = (4,4))
        for i, rank_loc in enumerate(rank_locations):
            row = rank_loc[0]
            col = rank_loc[1]
            rank = col*nrow + row
            plt.plot(var[rank][:outer_iter,0], var[rank][:outer_iter,n+1], label=f'({row},{col})', linestyle=linestyles[i//2], color=colors[i//2], 
                    linewidth=2, marker=markers[i%2], markersize=5, markevery=outer_iter//5)
        plt.xlabel('0uter Iterations', fontsize=14)
        if n == 0:
            plt.ylabel(r'$\alpha$', fontsize=14)
            plt.legend(prop={'size': 12}, frameon=False)
        elif n == 1:
            plt.ylabel(r'$\beta$', fontsize=14)
        else:
            plt.ylabel(r'$\gamma$', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{inter_name}_inter_para.png', dpi=300)
        
    # Plot outer iterations v.s. interface ratios
    inter_list = ['beta', 'gamma']
    for n, inter_name in enumerate(inter_list):
        fig = plt.figure(figsize = (4,4))
        for i, rank_loc in enumerate(rank_locations):
            row = rank_loc[0]
            col = rank_loc[1]
            rank = col*nrow + row
            plt.plot(var[rank][:outer_iter,0], var[rank][:outer_iter,n+2]/var[rank][:outer_iter,1], label=f'({row},{col})', linestyle=linestyles[i//2], color=colors[i//2], 
                    linewidth=2, marker=markers[i%2], markersize=5, markevery=outer_iter//5)
        plt.xlabel('0uter Iterations', fontsize=14)
        if n == 0:
            plt.ylabel(r'$\beta / \alpha$', fontsize=14)
        else:
            plt.ylabel(r'$\gamma/ \alpha$', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{inter_name}2alpha_ratio.png', dpi=300)

    # Plot outer iterations v.s. norms
    norm_list = ['l2', 'linf']
    for n, var in enumerate([l2_s, linf_s]):
        fig = plt.figure(figsize = (4,4))
        for i, rank_loc in enumerate(rank_locations):
            row = rank_loc[0]
            col = rank_loc[1]
            rank = col*nrow + row
            plt.plot(var[rank][:outer_iter,0], var[rank][:outer_iter,1], label=f'({row},{col})', linestyle=linestyles[i//2], color=colors[i//2], 
                    linewidth=2, marker=markers[i%2], markersize=5, markevery=outer_iter//5)
        plt.xlabel('Outer Iterations', fontsize=14)
        if n == 0:
            plt.ylabel(r'$\mathcal{E}_r$', fontsize=14)
        else:
            plt.ylabel(r'$\mathcal{E}_{\infty}$', fontsize=14)
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{norm_list[n]}.png', dpi=300)



