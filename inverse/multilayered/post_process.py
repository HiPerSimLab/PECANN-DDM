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
from matplotlib.lines import Line2D


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
        fig = plt.figure(figsize = (6,5))
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
                ax.title.set_text(fr'$\Omega_{rank}$')

        # Add colorbar to the right of the subplots
        cbar_ax = fig.add_subplot(gs[:, -1])
        ticks = np.linspace(vmin, vmax, 5, endpoint=True)
        cbar = fig.colorbar(pimg, cax=cbar_ax, ticks=ticks)
        limit = np.ceil(np.log10(vmax))-1
        cbar.formatter.set_powerlimits((limit, limit))
        cbar.update_ticks()
        plt.savefig(f'pic/{methodname}_{trial}_{varlist[k]}.png', dpi=300)

    linestyles = ['-.', '--', '-', ':']
    colors = ['b', 'r', 'k', 'g'] 
    markers = ['o', 'd', 's', '^']

    y_list = [0.6, 0.9]
    for k, y_idx in enumerate([77, 115]):
        y_label = y_sub[y_idx,0]
        print('y = ', y_label)
        fig = plt.figure(figsize = (5,4))
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                x_sub = x[:,rank].reshape(test_dis[1]+1, test_dis[0]+1)
                y_sub = y[:,rank].reshape(test_dis[1]+1, test_dis[0]+1)
                u_pred_sub = u_pred[:,rank].reshape(test_dis[1]+1, test_dis[0]+1)
                u_star_sub = u_star[:,rank].reshape(test_dis[1]+1, test_dis[0]+1)
        
                plt.plot(x_sub[y_idx,:], u_star_sub[y_idx,:], label='Truth' if i==0 else '',
                        linestyle='-', color='k', linewidth=2)
                plt.plot(x_sub[y_idx,:], u_pred_sub[y_idx,:], label=fr'$\Omega_{rank}$',
                        linestyle=linestyles[i], color=colors[i], linewidth=2, 
                         marker=markers[i], markersize=5, markevery=20)
        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(fr'$u(x,y={y_list[k]})$', fontsize=14)
        if k == 1:
            plt.legend(prop={'size': 12}, frameon=False)
        plt.tight_layout()
        plt.ylim([-0.2, 10.2])
        plt.savefig(f'pic/{methodname}_{trial}_y_{y_list[k]}.png', dpi=300)


#


def scatter_points(trial, dims, full_domain, methodname):
    nx, ny   = dims
    x_joints = np.linspace(full_domain[0,0], full_domain[1,0], nx+1)
    y_joints = np.linspace(full_domain[0,1], full_domain[1,1], ny+1)

    fig = plt.figure(figsize = (7,5)) #(x,y): 1D: (9,5); 2D: (6,5)
    gs = gridspec.GridSpec(ny, nx + 1, width_ratios=[1]*nx + [0.2], wspace=0.1, hspace=0.1)

    for i in range(nx):
        for j in range(ny):
            rank = i*ny + j
            ax = plt.subplot(gs[j,i])
            data_dom = np.loadtxt(f"data/{trial}_{rank}_dom.dat")
            data_bc  = np.loadtxt(f"data/{trial}_{rank}_bc.dat")
            data_int = np.loadtxt(f"data/{trial}_{rank}_int.dat")
            data_da = np.loadtxt(f"data/{trial}_{rank}_da.dat")
            plt.scatter(data_dom[:,0], data_dom[:,1], s=1, color='k', marker='o',
                        label='Residual' if i == 1 and j == 0 else "")
            if np.isnan(data_int).any() == False:
                plt.scatter(data_int[:,0], data_int[:,1], s=8, color='b', marker='d',
                           label='Interface' if i == 1 and j == 0 else "")
            if np.isnan(data_bc).any() == False:
                plt.scatter(data_bc[:,0], data_bc[:,1], s=16, color='g', marker='^',
                            label='Boundary' if i == 1 and j == 0 else "")
            plt.scatter(data_da[:,0], data_da[:,1], s=32, color='r', marker='x',
                       label='Data' if i == 1 and j == 0 else "")

            if i == 0:
                yticks = np.linspace(y_joints[-j-2], y_joints[-j-1], 3).tolist()
                ax.set_yticks(yticks)
            if j == ny-1:
                xticks = np.linspace(x_joints[i], x_joints[i+1], 2).tolist()
                ax.set_xticks(xticks)
            ax.label_outer()
            ax.axis('scaled')
            ax.title.set_text(fr'$\Omega_{rank}$')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Residual', markerfacecolor='k', markersize=8),
        Line2D([0], [0], marker='d', color='w', label='Interface', markerfacecolor='b', markersize=8),
        Line2D([0], [0], marker='^', color='w', label='Boundary', markerfacecolor='g', markersize=8),
        Line2D([0], [0], marker='x', color='w', label='Data', markeredgecolor='r', markersize=8)
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.0, 0.5), prop={'size': 12}, frameon=False)

    plt.savefig(f'pic/{methodname}_{trial}_points.png', dpi=300)


# In[102]:


def plot_update(trial, dims, size, outer_iter, methodname):
    obj_s = []
    mu_s = []
    constr_s = []
    lambda_s = []
    q_s = []
    l2_s = []
    linf_s = []
    k_s = []
    for rank in range(size):
        obj_s.append( np.loadtxt(f'data/{trial}_{rank}_object.dat') )
        mu_s.append( np.loadtxt(f'data/{trial}_{rank}_mu.dat') )
        constr_s.append( np.loadtxt(f'data/{trial}_{rank}_constr.dat') )
        lambda_s.append( np.loadtxt(f'data/{trial}_{rank}_lambda.dat') )
        q_s.append( np.loadtxt(f'data/{trial}_{rank}_q.dat') )
        l2_s.append( np.loadtxt(f'data/{trial}_{rank}_l2.dat') )
        linf_s.append( np.loadtxt(f'data/{trial}_{rank}_linf.dat') )
        k_s.append( np.loadtxt(f'data/{trial}_{rank}_k.dat') )

    linestyles = ['-.', '--', '-', ':']
    colors = ['b', 'r', 'k', 'g'] 
    markers = ['o', 'd', 's', '^'] 

    nx, ny   = dims

    # Plot outer iterations v.s. objectives
    fig = plt.figure(figsize = (4,4))
    plt.plot(obj_s[0][:outer_iter,0], obj_s[0][:outer_iter,1], label=r'$\mathcal{J}(\Omega_1)$', linestyle=linestyles[2], color=colors[0], linewidth=2, 
             marker=markers[0], markersize=5, markevery=outer_iter//5)
    plt.plot(obj_s[1][:outer_iter,0], obj_s[1][:outer_iter,1], label=r'$\mathcal{J}(\Omega_2)$', linestyle=linestyles[2], color=colors[1], linewidth=2, 
             marker=markers[0], markersize=5, markevery=outer_iter//5)
    plt.xlabel('Outer Iterations', fontsize=14)
    plt.ylabel(r'$\mathcal{J}$', fontsize=14)
    plt.semilogy()
    plt.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.savefig(f'pic/{methodname}_{trial}_objective.png', dpi=300)

    # Plot outer iterations v.s. constraints
    var_list = 'constr'
    var = constr_s
    fig = plt.figure(figsize = (4,4))
    
    constr_list = ['BC', 'Data', 'PDE']
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,1], label=r'$\mathcal{C}_{B}(\partial \Omega_1)$', linestyle=linestyles[0], color=colors[0], linewidth=2., 
                         marker=markers[1], markersize=5, markevery=outer_iter//5)
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,3], label=r'$\mathcal{C}_{D}(\mathcal{M}_1)$', linestyle=linestyles[1], color=colors[0], linewidth=2., 
                         marker=markers[2], markersize=5, markevery=outer_iter//5)
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,2], label=r'$\mathcal{C}_{P}(\Omega_1)$', linestyle=linestyles[3], color=colors[0], linewidth=2., 
                         marker=markers[3], markersize=5, markevery=outer_iter//5)
    
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,1], label=r'$\mathcal{C}_{B}(\partial \Omega_2)$', linestyle=linestyles[0], color=colors[1], linewidth=2., 
                          marker=markers[1], markersize=5, markevery=outer_iter//5)
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,3], label=r'$\mathcal{C}_{D}(\mathcal{M}_2)$', linestyle=linestyles[1], color=colors[1], linewidth=2., 
                           marker=markers[2], markersize=5, markevery=outer_iter//5)
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,2], label=r'$\mathcal{C}_{P}(\Omega_2)$', linestyle=linestyles[3], color=colors[1], linewidth=2., 
                         marker=markers[3], markersize=5, markevery=outer_iter//5)    

    plt.xlabel('Outer Iterations', fontsize=14)
    plt.ylabel(r'$\mathcal{C}$', fontsize=14)
    plt.semilogy()
    plt.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.savefig(f'pic/{methodname}_{trial}_{var_list}_loss.png', dpi=300)

    # Plot outer iterations v.s. Lagrange multipliers
    var_list = 'lambda'
    var = lambda_s
    fig = plt.figure(figsize = (4,4))

    constr_list = ['BC', 'Data', 'PDE']
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,1], label=r'$\lambda_{B}(\partial \Omega_1)$', linestyle=linestyles[0], color=colors[0], linewidth=2.,
                         marker=markers[1], markersize=5, markevery=outer_iter//5)
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,3], label=r'$\lambda_{D}(\mathcal{M}_1)$', linestyle=linestyles[1], color=colors[0], linewidth=2.,
                         marker=markers[2], markersize=5, markevery=outer_iter//5)
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,2], label=r'$\lambda_{P}(\Omega_1)$', linestyle=linestyles[3], color=colors[0], linewidth=2.,
                         marker=markers[3], markersize=5, markevery=outer_iter//5)

    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,1], label=r'$\lambda_{B}(\partial \Omega_2)$', linestyle=linestyles[0], color=colors[1], linewidth=2.,
                          marker=markers[1], markersize=5, markevery=outer_iter//5)
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,3], label=r'$\lambda_{D}(\mathcal{M}_2)$', linestyle=linestyles[1], color=colors[1], linewidth=2., 
                           marker=markers[2], markersize=5, markevery=outer_iter//5)
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,2], label=r'$\lambda_{P}(\Omega_2)$', linestyle=linestyles[3], color=colors[1], linewidth=2., 
                         marker=markers[3], markersize=5, markevery=outer_iter//5)

    plt.xlabel('Outer Iterations', fontsize=14)
    plt.ylabel(r'$\lambda$', fontsize=14)
    plt.semilogy()
    plt.legend(prop={'size': 12}, frameon=False)
    plt.ylim([1e-1, 1e+6])
    plt.tight_layout()
    plt.savefig(f'pic/{methodname}_{trial}_{var_list}.png', dpi=300)

    # Plot outer iterations v.s. penalty parameters
    var_list = 'mu'
    var = mu_s
    fig = plt.figure(figsize = (4,4))

    constr_list = ['BC', 'Data', 'PDE']
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,1], label=r'$\mu_{B}(\partial \Omega_1)$', linestyle=linestyles[0], color=colors[0], linewidth=2.,
                         marker=markers[1], markersize=5, markevery=outer_iter//5)
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,3], label=r'$\mu_{D}(\mathcal{M}_1)$', linestyle=linestyles[1], color=colors[0], linewidth=2.,
                         marker=markers[2], markersize=5, markevery=outer_iter//5)
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,2], label=r'$\mu_{P}(\Omega_1)$', linestyle=linestyles[3], color=colors[0], linewidth=2.,
                         marker=markers[3], markersize=5, markevery=outer_iter//5)

    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,1], label=r'$\mu_{B}(\partial \Omega_2)$', linestyle=linestyles[0], color=colors[1], linewidth=2.,
                          marker=markers[1], markersize=5, markevery=outer_iter//5)
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,3], label=r'$\mu_{D}(\mathcal{M}_2)$', linestyle=linestyles[1], color=colors[1], linewidth=2.,
                           marker=markers[2], markersize=5, markevery=outer_iter//5)
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,2], label=r'$\mu_{P}(\Omega_2)$', linestyle=linestyles[3], color=colors[1], linewidth=2.,
                         marker=markers[3], markersize=5, markevery=outer_iter//5)

    plt.xlabel('Outer Iterations', fontsize=14)
    plt.ylabel(r'$\mu$', fontsize=14)
    plt.semilogy()
    plt.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.savefig(f'pic/{methodname}_{trial}_{var_list}.png', dpi=300)
    
    # Plot outer iterations v.s. interface parameters
    robin_list = ['alpha', 'beta', 'gamma']
    var = q_s
    var_list = 'intfac_para'
    fig = plt.figure(figsize = (4,4))
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,1], label=r'$\alpha(\Omega_1)$', linestyle=linestyles[0], color=colors[0], linewidth=2., 
                         marker=markers[0], markersize=5, markevery=outer_iter//5)
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,2], label=r'$\beta(\Omega_1)$', linestyle=linestyles[1], color=colors[0], linewidth=2., 
                         marker=markers[1], markersize=5, markevery=outer_iter//5)
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,3], label=r'$\gamma(\Omega_1)$', linestyle=linestyles[2], color=colors[0], linewidth=2., 
                         marker=markers[2], markersize=5, markevery=outer_iter//5)
    
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,1], label=r'$\alpha(\Omega_2)$', linestyle=linestyles[0], color=colors[1], linewidth=2., 
                         marker=markers[0], markersize=5, markevery=outer_iter//5)
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,2], label=r'$\beta(\Omega_2)$', linestyle=linestyles[1], color=colors[1], linewidth=2., 
                         marker=markers[1], markersize=5, markevery=outer_iter//5)
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,3], label=r'$\gamma(\Omega_2)$', linestyle=linestyles[2], color=colors[1], linewidth=2., 
                         marker=markers[2], markersize=5, markevery=outer_iter//5)
    
    plt.xlabel('Outer Iterations', fontsize=14)
    plt.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.savefig(f'pic/{methodname}_{trial}_{var_list}.png', dpi=300)

    # Plot outer iterations v.s. interface ratios
    var_list = 'intfac_ratio'
    fig = plt.figure(figsize = (4,4))
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,2]/var[0][:outer_iter,1], label=r'$\beta/\alpha$'+' '+ r'$(\Omega_1)$', linestyle=linestyles[1], color=colors[0], linewidth=2., 
                         marker=markers[1], markersize=5, markevery=outer_iter//5)
    plt.plot(var[0][:outer_iter,0], var[0][:outer_iter,3]/var[0][:outer_iter,1], label=r'$\gamma/\alpha$'+' '+ r'$(\Omega_1)$', linestyle=linestyles[2], color=colors[0], linewidth=2., 
                         marker=markers[2], markersize=5, markevery=outer_iter//5)
    
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,2]/var[0][:outer_iter,1], label=r'$\beta/\alpha$'+' '+ r'$(\Omega_2)$', linestyle=linestyles[1], color=colors[1], linewidth=2., 
                         marker=markers[1], markersize=5, markevery=outer_iter//5)
    plt.plot(var[1][:outer_iter,0], var[1][:outer_iter,3]/var[0][:outer_iter,1], label=r'$\gamma/\alpha$'+' '+ r'$(\Omega_2)$', linestyle=linestyles[2], color=colors[1], linewidth=2., 
                         marker=markers[2], markersize=5, markevery=outer_iter//5)
    
    plt.xlabel('Outer Iterations', fontsize=14)
    plt.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.savefig(f'pic/{methodname}_{trial}_{var_list}.png', dpi=300)

    # Plot outer iterations v.s. norms
    # temperature
    var_list = 'u'
    norm_list = ['l2', 'linf']
    for n, norm in enumerate([l2_s, linf_s]):
        fig = plt.figure(figsize = (4,4))
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                plt.plot(norm[rank][:outer_iter,0], norm[rank][:outer_iter,1], label=fr'$\Omega_{rank}$', linestyle=linestyles[i], color=colors[i], linewidth=2, 
                         marker=markers[i], markersize=5, markevery=outer_iter//5)
        plt.xlabel('Outer Iterations', fontsize=14)
        if n == 0:
            plt.ylabel(r'$\mathcal{E}_r^{(u)}$', fontsize=14)
        else:
            plt.ylabel(r'$\mathcal{E}_{\infty}^{(u)}$', fontsize=14)
        plt.semilogy()
        plt.legend(prop={'size': 12}, frameon=False)
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{norm_list[n]}_{var_list}.png', dpi=300)

    # conductivity
    var_list = 'k'
    for n, norm in enumerate([k_s]):
        fig = plt.figure(figsize = (4,4))
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                plt.plot(norm[rank][:outer_iter,0], norm[rank][:outer_iter,1], label=fr'$\Omega_{rank}$', 
                         linestyle=linestyles[i], color=colors[i], linewidth=2, marker=markers[i], markersize=5, markevery=outer_iter//5)
        plt.xlabel('Outer Iterations', fontsize=14)
        plt.ylabel(r'$k$', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{var_list}.png', dpi=300)

    var_list = 'k'
    norm_list = ['linf']
    for n, norm in enumerate([linf_s]):
        fig = plt.figure(figsize = (4,4))
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                plt.plot(norm[rank][:outer_iter,0], norm[rank][:outer_iter,2], label=fr'$\Omega_{rank}$', linestyle=linestyles[i], color=colors[i], linewidth=2,
                         marker=markers[i], markersize=5, markevery=outer_iter//5)
        plt.xlabel('Outer Iterations', fontsize=14)
        plt.ylabel(r'$\mathcal{E}_{\infty}^{(u)}$', fontsize=14)
        plt.semilogy()
        plt.legend(prop={'size': 12}, frameon=False)
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{norm_list[n]}_{var_list}.png', dpi=300)

