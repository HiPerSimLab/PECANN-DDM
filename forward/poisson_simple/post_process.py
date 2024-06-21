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

from sample import rank_domain, fetch_interior_data


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



def contour_prediction(test_dis, trial, methodname):    
    u_up = np.loadtxt(f'./data/{methodname}_{trial}_x_y_u_upred.dat')
    
    x      = u_up[:,0]
    y      = u_up[:,1]
    u_star = u_up[:,2]
    u_pred = u_up[:,3]
    u_err  = abs(u_star - u_pred)
    
    varlist = ['u_star', 'u_pred', 'u_err']
    for k, var in enumerate([u_star, u_pred, u_err]):
        fig = plt.figure(figsize = (6,5)) #(x,y): 1D: (9,5); 2D: (6,5)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1] + [0.1], wspace=0., hspace=0.4)

        x_sub = x.reshape(test_dis[1]+1, test_dis[0]+1)
        y_sub = y.reshape(test_dis[1]+1, test_dis[0]+1)
        var_sub = var.reshape(test_dis[1]+1, test_dis[0]+1)
        if k == 2:
            vmax = np.max(var)
            vmin = np.min(var)
        else:
            vmax = np.max(u_star)
            vmin = np.min(u_star)
        #vmax = np.max(var)
        #vmin = np.min(var)

        ax = plt.subplot(gs[0,0])
        pimg=plt.pcolormesh(x_sub ,y_sub ,var_sub,vmin=vmin, vmax=vmax, cmap='jet', shading='gouraud')

        ax.label_outer()
        ax.axis('scaled')

        # Add colorbar to the right of the subplots
        cbar_ax = fig.add_subplot(gs[:, -1])
        ticks = np.linspace(vmin, vmax, 5, endpoint=True)
        cbar = fig.colorbar(pimg, cax=cbar_ax, ticks=ticks)
        limit = np.ceil(vmax)-1
        cbar.formatter.set_powerlimits((limit, limit))
        cbar.update_ticks()

        #plt.title(f'w = {omega.cpu().numpy().tolist()}')
        plt.savefig(f'pic/{methodname}_{trial}_{varlist[k]}.png', dpi=300)
        plt.close()



# In[102]:


def sample_residual_distribution(omega, test_dis, trial, dims, size, methodname):
    xy_pde_loss = np.loadtxt(f'./data/{methodname}_{trial}_x_y_pde_loss.dat')

    n        = xy_pde_loss.shape[0] // size
    x        = xy_pde_loss[:,0].reshape(size, n).T
    y        = xy_pde_loss[:,1].reshape(size, n).T
    pde_loss = xy_pde_loss[:,2].reshape(size, n).T

    ### Follow MPI Cart Topology ###
    nx, ny   = dims
    x_joints = np.linspace(x.min(), x.max(), nx+1)
    y_joints = np.linspace(y.min(), y.max(), ny+1)


    varlist = ['pde_loss']
    for k, var in enumerate([pde_loss]):
        fig = plt.figure(figsize = (6,5)) #(x,y): 1D: (9,5); 2D: (6,5)
        gs = gridspec.GridSpec(ny, nx+1, wspace=0.4, hspace=0.4)
        
        vmax = 1e-2 #np.max(var)
        vmin = 1e-6 #np.min(var)
        norm = LogNorm(vmin = vmin, vmax = vmax)
        
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                x_sub = x[:,rank].reshape(test_dis[1]+1, test_dis[0]+1)
                y_sub = y[:,rank].reshape(test_dis[1]+1, test_dis[0]+1)
                var_sub = var[:,rank].reshape(test_dis[1]+1, test_dis[0]+1)

                ax = plt.subplot(gs[j,i])
                pimg=plt.pcolormesh(x_sub ,y_sub ,var_sub, norm=norm, cmap='jet', shading='gouraud')
                #data_dom = np.loadtxt(f"data/{trial}_{rank}_dom.dat")
                #data_bc  = np.loadtxt(f"data/{trial}_{rank}_bc.dat")
                #data_int = np.loadtxt(f"data/{trial}_{rank}_int.dat")
                #plt.scatter(data_dom[:,0], data_dom[:,1], s=2)
                #if np.isnan(data_bc).any() == False:
                #    plt.scatter(data_bc[:,0], data_bc[:,1], s=2)
                #if np.isnan(data_int).any() == False:
                #    plt.scatter(data_int[:,0], data_int[:,1], s=2)

                if i == 0:
                    yticks = np.linspace(y_joints[-j-2], y_joints[-j-1], 2).tolist()
                    ax.set_yticks(yticks)
                if j == ny-1:
                    xticks = np.linspace(x_joints[i], x_joints[i+1], 2).tolist()
                    ax.set_xticks(xticks)
                ax.label_outer()
                ax.axis('scaled')
                ax.title.set_text(f'({i},{j})')
        
        mappable = ScalarMappable(norm=norm, cmap='jet')
        cbar_position = [0.8, 0.1, 0.03, 0.8]  # [left, bottom, width, height]
        cbar_ax = fig.add_axes(cbar_position)
        cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5) #adjust 'num' as needed
        cbar = fig.colorbar(pimg, cax=cbar_ax, ticks=cbar_ticks) 
        cbar.ax.set_yticklabels(["{:.1e}".format(tick) for tick in cbar_ticks], fontsize=8)
        
        #plt.title(f'w = {omega.cpu().numpy().tolist()}')
        plt.savefig(f'pic/{methodname}_{trial}_{varlist[k]}.png', dpi=300)
        plt.close()



# In[102]:


def plot_update(trial, dims, size, outer_iter, epochs, methodname):
    obj_s = []
    mu_s = []
    constr_s = []
    lambda_s = []
    alpha_s = []
    l2_s = []
    linf_s = []
    for rank in range(size):
        obj_s.append( np.loadtxt(f'data/{trial}_{rank}_{outer_iter}_object.dat') )
        mu_s.append( np.loadtxt(f'data/{trial}_{rank}_{outer_iter}_mu.dat') )
        constr_s.append( np.loadtxt(f'data/{trial}_{rank}_{outer_iter}_constr.dat') )
        lambda_s.append( np.loadtxt(f'data/{trial}_{rank}_{outer_iter}_lambda.dat') )
        alpha_s.append( np.loadtxt(f'data/{trial}_{rank}_{outer_iter}_alpha.dat') )
        l2_s.append( np.loadtxt(f'data/{trial}_{rank}_{outer_iter}_l2.dat') )
        linf_s.append( np.loadtxt(f'data/{trial}_{rank}_{outer_iter}_linf.dat') )

    linestyles = ['-', '--', '-.', ':']
    colors = ['b', 'k', 'g', 'r']  # Blue, Black, Green, Red
    markers = ['o', 's', '^', 'd']  # Circle, Square, Triangle up, Diamond

    fig = plt.figure(figsize = (4,4))
    nx, ny = dims
    for i in range(nx):
        for j in range(ny):
            rank = i*ny + j
            plt.plot(obj_s[rank][:,0], obj_s[rank][:,1], label=f'({i},{j})', linestyle=linestyles[i], color=colors[i], linewidth=2, marker=markers[j], markersize=5, markevery=outer_iter//10)
    plt.xlabel('Outer Iterations', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('objective: averaging interface loss', fontsize=14)
    plt.semilogy()
    plt.legend(prop={'size': 12}, frameon=False)
    #plt.grid()
    plt.tight_layout()
    plt.savefig(f'pic/{methodname}_{trial}_objective.png', dpi=300)
    plt.close()

    var_list = ['mu', 'constr', 'lambda']
    for k, var in enumerate([mu_s, constr_s, lambda_s]):
        constr_list = ['Boundary', 'PDE']
        for n, constr_name in enumerate(constr_list):
            fig = plt.figure(figsize = (4,4))
            for i in range(nx):
                for j in range(ny):
                    rank = i*ny + j
                    plt.plot(var[rank][:,0], var[rank][:,n+1], label=f'({i},{j})', linestyle=linestyles[i], color=colors[i], linewidth=2, marker=markers[j], markersize=5, markevery=outer_iter//10)
            plt.xlabel('Outer Iterations', fontsize=12)
            plt.ylabel(var_list[k], fontsize=12)
            plt.title(f'{var_list[k]} update of the {constr_name} constraint', fontsize=14)
            plt.semilogy()
            plt.legend(prop={'size': 12}, frameon=False)
            #plt.grid()
            plt.tight_layout()
            plt.savefig(f'pic/{methodname}_{trial}_{var_list[k]}_{constr_name}_constr.png', dpi=300)
            plt.close()

    robin_list = ['alpha', 'beta', 'gamma']
    for n, robin_name in enumerate(robin_list):
        fig = plt.figure(figsize = (4,4))
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                if n == 2:
                    plt.plot(alpha_s[rank][:,0], alpha_s[rank][:,-1]/alpha_s[rank][:,-2], label=f'({i},{j})', linestyle=linestyles[i], color=colors[i], linewidth=2, marker=markers[j], markersize=5, markevery=outer_iter//10)
                else:
                    plt.plot(alpha_s[rank][:,0], alpha_s[rank][:,n+1], label=f'({i},{j})', linestyle=linestyles[i], color=colors[i], linewidth=2, marker=markers[j], markersize=5, markevery=outer_iter//10)
        plt.xlabel('0uter Iterations', fontsize=12)
        plt.ylabel(robin_name, fontsize=12)
        plt.title(f'{robin_name} update of Robin interface condition', fontsize=14)
        plt.legend(prop={'size': 12}, frameon=False)
        #plt.grid()
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{robin_name}_robin_interface.png', dpi=300)
        plt.close()

    norm_list = ['l2', 'linf']
    for n, norm in enumerate([l2_s, linf_s]):
        fig = plt.figure(figsize = (4,4))
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                plt.plot(norm[rank][:,0], norm[rank][:,1], label=f'({i},{j})', linestyle=linestyles[i], color=colors[i], linewidth=2, marker=markers[j], markersize=5, markevery=outer_iter//10)
        plt.xlabel('Outer Iterations', fontsize=12)
        plt.ylabel(norm_list[n], fontsize=12)
        plt.title(f'{norm_list[n]} update', fontsize=14)
        plt.semilogy()
        plt.legend(prop={'size': 12}, frameon=False)
        #plt.grid()
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{norm_list[n]}.png', dpi=300)
        plt.close()


# In[134]:


def plot_exchange(trial, dims, size, outer_iter, methodname):
    u_adj_s  = []
    du_adj_s = []
    for rank in range(size):
        u_adj_s.append( np.loadtxt(f'data/{trial}_{rank}_{outer_iter}_u_adj.dat') )
        du_adj_s.append( np.loadtxt(f'data/{trial}_{rank}_{outer_iter}_du_adj.dat') )

    n = u_adj_s[0].shape[1]
    outer = np.arange(1, outer_iter+1)
    
    y = np.random.randint(n)
    var_list = ['value', 'flux']
    for j, var in enumerate([u_adj_s, du_adj_s]):
        fig = plt.figure(figsize = (9,6))
        for rank in range(size):
            if rank != 0 and rank != size-1:
                var_sub = var[rank][:,:n]
                plt.plot(var_sub[:,y], label=f'rank {rank}')
                
                var_sub = var[rank][:,n:]
                plt.plot(var_sub[:,y], label=f'rank {rank}')
            else:
                var_sub = var[rank]
                plt.plot(var_sub[:,y], label=f'rank {rank}')
        plt.xlabel('outer iteration')
        plt.ylabel(f'{var_list[j]}')
        plt.title(f'{var_list[j]} constraint at the interface of random y index {y}')
        plt.legend()
        plt.grid()
        plt.savefig(f'pic/{methodname}_{trial}_{var_list[j]}_exchange.png')
        plt.close()


# In[ ]:




