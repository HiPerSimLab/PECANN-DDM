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


def boundary_location(theta):
    rho       = 1 + np.cos(theta)*np.sin(4*theta)
    radius    = rho
    x         = 6*0.55*radius*np.cos(theta)
    y         = 6*0.75*radius*np.sin(theta)
    return x,y



def contour_prediction(trial, methodname):
    u_up = np.loadtxt(f'./data/{methodname}_{trial}_x_y_u_upred.dat')

    x      = u_up[:,0]
    y      = u_up[:,1]
    u_star = u_up[:,2]
    u_pred = u_up[:,3]
    u_err  = abs(u_star - u_pred)
    k_star = u_up[:,4]
    k_pred = u_up[:,5]
    k_err  = abs(k_star - k_pred)

    theta = np.linspace(0, 2*np.pi, 1000)
    bound_x, bound_y = boundary_location(theta)
    
    varlist = ['u_star', 'u_pred', 'u_err']
    for k, var in enumerate([u_star, u_pred, u_err]):
        fig = plt.figure(figsize = (6,5))
        gs = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.4)

        if k == 2:
            vmax = np.max(var)
            vmin = np.min(var)
        else:
            vmax = np.max(u_star)
            vmin = np.min(u_star)

        ax = plt.subplot(gs[0,0])
        plt.plot(bound_x, bound_y, 'k-', linewidth=1)
        pimg=plt.scatter(x,y,c=var,s=2,vmin=vmin, vmax=vmax, cmap=cmap, edgecolors='face')
        plt.vlines(x=0., ymin=boundary_location(-np.pi/2)[1], ymax=boundary_location(np.pi/2)[1],
                   colors='grey', linestyles=':', lw=1)
        plt.hlines(y=0., xmin=boundary_location(np.pi)[0], xmax=boundary_location(0.)[0],
                   colors='grey', linestyles=':', lw=1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.axis('square')
        limit = np.ceil(np.log10(vmax))-1
        colorbar(pimg,vmin,vmax,limit)

        plt.savefig(f'pic/{methodname}_{trial}_{varlist[k]}.png', dpi=300)
        
    varlist = ['k_star', 'k_pred', 'k_err']
    for k, var in enumerate([k_star, k_pred, k_err]):
        fig = plt.figure(figsize = (6,5)) #(x,y): 1D: (9,5); 2D: (6,5)
        gs = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.4)

        if k == 2:
            vmax = np.max(var)
            vmin = np.min(var)
        else:
            vmax = np.max(k_star)
            vmin = np.min(k_star)

        ax = plt.subplot(gs[0,0])
        plt.plot(bound_x, bound_y, 'k-', linewidth=1)
        pimg=plt.scatter(x,y,c=var,s=2,vmin=vmin, vmax=vmax, cmap=cmap, edgecolors='face')
        plt.vlines(x=0., ymin=boundary_location(-np.pi/2)[1], ymax=boundary_location(np.pi/2)[1],
                   colors='grey', linestyles=':', lw=1)
        plt.hlines(y=0., xmin=boundary_location(np.pi)[0], xmax=boundary_location(0.)[0],
                   colors='grey', linestyles=':', lw=1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.axis('square')
        limit = np.ceil(np.log10(vmax))-1
        colorbar(pimg,vmin,vmax,limit)

        plt.savefig(f'pic/{methodname}_{trial}_{varlist[k]}.png', dpi=300)


#


def scatter_points(trial, dims, methodname):
    fig = plt.figure(figsize = (5,5))
    gs = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.4)
    ax = plt.subplot(gs[0,0])
    theta = np.linspace(0, 2*np.pi, 1000)
    bound_x, bound_y = boundary_location(theta)
    plt.plot(bound_x, bound_y, 'k-', linewidth=1)

    nx, ny   = dims
    for i in range(nx):
        for j in range(ny):
            rank = i*ny + j
            data_dom = np.loadtxt(f"data/{trial}_{rank}_dom.dat")
            data_int = np.loadtxt(f"data/{trial}_{rank}_int.dat")
            scatter1 = plt.scatter(data_dom[:,0], data_dom[:,1], s=1, color='k', marker='o',
                                  label = 'Residual' if i == nx-1 and j== ny-1 else '')
            if np.isnan(data_int).any() == False:
                scatter2 = plt.scatter(data_int[:,0], data_int[:,1], s=2, color='b', marker='d',
                                      label = 'Interface' if i == nx-1 and j== ny-1 else '')
    plt.text(-2.5, 3.6, '(0, 0)')
    plt.text(-3.5, -2, '(1, 0)')
    plt.text(1, 3.6, '(0, 1)')
    plt.text(2, -2, '(1, 1)')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Residual', markerfacecolor='k', markersize=8),
        Line2D([0], [0], marker='d', color='w', label='Interface', markerfacecolor='b', markersize=8)
    ]

    ax.legend(handles=legend_elements, prop={'size': 12}, frameon=False)
    ax.axis('square')

    plt.savefig(f'pic/{methodname}_{trial}_points_dom.png', dpi=300)

    fig = plt.figure(figsize = (5,5))
    gs = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.4)
    ax = plt.subplot(gs[0,0])
    nx, ny   = dims
    for i in range(nx):
        for j in range(ny):
            rank = i*ny + j
            data_da  = np.loadtxt(f"data/{trial}_{rank}_da.dat")
            data_bc  = np.loadtxt(f"data/{trial}_{rank}_bc.dat")
            plt.scatter(data_da[:,0], data_da[:,1], s=32, color='r', marker='x')
            if np.isnan(data_bc).any() == False:
                plt.scatter(data_bc[:,0], data_bc[:,1], s=32, color='g', marker='^')
    plt.text(-2.5, 3.6, '(0, 0)')
    plt.text(-3.5, -2, '(1, 0)')
    plt.text(1, 3.6, '(0, 1)')
    plt.text(2, -2, '(1, 1)')
    plt.vlines(x=0., ymin=boundary_location(-np.pi/2)[1], ymax=boundary_location(np.pi/2)[1],
               colors='grey', linestyles='--', lw=2)
    plt.hlines(y=0., xmin=boundary_location(np.pi)[0], xmax=boundary_location(0.)[0],
               colors='grey', linestyles='--', lw=2)
    theta = np.linspace(0, 2*np.pi, 1000)
    bound_x, bound_y = boundary_location(theta)
    plt.plot(bound_x, bound_y, 'k-', linewidth=1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(['Data', 'Boundary'],prop={'size': 12}, frameon=False)
    ax.axis('square')

    plt.savefig(f'pic/{methodname}_{trial}_points_data.png', dpi=300)


# In[102]:


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

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
        
    linestyles = ['-', '--', '-.', ':']
    colors = ['b', 'k', 'g', 'r']  # Blue, Black, Green, Red
    markers = ['o', 's', '^', 'd']  # Circle, Square, Triangle up, Diamond
    
    nx, ny   = dims
    
    fig = plt.figure(figsize = (4,4))
    k = 0
    for i in range(nx):
        for j in range(ny):
            rank = i*ny + j
            mov_avg = moving_average(obj_s[rank][:,1], 10)
            plt.plot(mov_avg, label=f'({j},{i})', linestyle=linestyles[k], color=colors[k], linewidth=2, 
                     marker=markers[k], markersize=5, markevery=outer_iter//5)
            k += 1
    plt.xlabel('Outer Iterations', fontsize=14)
    plt.ylabel(r'$\mathcal{J}$', fontsize=14)
    plt.semilogy()
    plt.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.savefig(f'pic/{methodname}_{trial}_objective.png', dpi=300)

    var_list = 'constr'
    var = constr_s
    constr_list = ['BC', 'PDE', 'Data']
    for n, constr_name in enumerate(constr_list):
        fig = plt.figure(figsize = (4,4))
        k = 0
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                mov_avg = moving_average(var[rank][:,n+1], 10)
                plt.plot(mov_avg, label=f'({j},{i})', linestyle=linestyles[k], color=colors[k], linewidth=2, 
                         marker=markers[k], markersize=5, markevery=outer_iter//5)
                k += 1
        plt.xlabel('Outer Iterations', fontsize=14)
        if constr_name == 'BC':
            plt.ylabel(r'$\mathcal{C}_{B}$', fontsize=14)
        elif constr_name == 'PDE':
            plt.ylabel(r'$\mathcal{C}_{P}$', fontsize=14)
        elif constr_name == 'Data':
            plt.ylabel(r'$\mathcal{C}_{D}$', fontsize=14)
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{var_list}_{constr_name}.png', dpi=300)
    
    var_list = 'lambda'
    var = lambda_s
    constr_list = ['BC', 'PDE', 'Data']
    for n, constr_name in enumerate(constr_list):
        fig = plt.figure(figsize = (4,4))
        k = 0
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                mov_avg = moving_average(var[rank][:,n+1], 10)
                plt.plot(var[rank][:,n+1], label=f'({j},{i})', linestyle=linestyles[k], color=colors[k], linewidth=2, 
                         marker=markers[k], markersize=5, markevery=outer_iter//5)
                k += 1
        plt.xlabel('Outer Iterations', fontsize=14)
        if constr_name == 'BC':
            plt.ylabel(r'$\lambda_{B}$', fontsize=14)
        elif constr_name == 'PDE':
            plt.ylabel(r'$\lambda_{P}$', fontsize=14)
        elif constr_name == 'Data':
            plt.ylabel(r'$\lambda_{D}$', fontsize=14)
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{var_list}_{constr_name}.png', dpi=300)

    var_list = 'mu'
    var = mu_s
    constr_list = ['BC', 'PDE', 'Data']
    for n, constr_name in enumerate(constr_list):
        fig = plt.figure(figsize = (4,4))
        k = 0
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                mov_avg = moving_average(var[rank][:,n+1], 10)
                plt.plot(var[rank][:,n+1], label=f'({j},{i})', linestyle=linestyles[k], color=colors[k], linewidth=2, 
                         marker=markers[k], markersize=5, markevery=outer_iter//5)
                k += 1
        plt.xlabel('Outer Iterations', fontsize=14)
        if constr_name == 'BC':
            plt.ylabel(r'$\mu_{B}$', fontsize=14)
        elif constr_name == 'PDE':
            plt.ylabel(r'$\mu_{P}$', fontsize=14)
        elif constr_name == 'Data':
            plt.ylabel(r'$\mu_{D}$', fontsize=14)
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{var_list}_{constr_name}.png', dpi=300)

    # temperature
    var_list = 't'
    norm_list = ['l2', 'linf']
    for n, norm in enumerate([l2_s, linf_s]):
        fig = plt.figure(figsize = (4,4))
        k = 0
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                mov_avg = moving_average(norm[rank][:,1], 10)
                plt.plot(mov_avg, label=f'({j},{i})', linestyle=linestyles[k], color=colors[k], linewidth=2, 
                         marker=markers[k], markersize=5, markevery=outer_iter//5)
                k += 1
        plt.xlabel('Outer Iterations', fontsize=14)
        if n == 0:
            plt.ylabel(r'$\mathcal{E}_r^{(u)}$', fontsize=14)
        else:
            plt.ylabel(r'$\mathcal{E}_{\infty}^{(u)}$', fontsize=14)
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{norm_list[n]}_{var_list}.png', dpi=300)
        
    # conductivity
    var_list = 'k'
    norm_list = ['l2', 'linf']
    for n, norm in enumerate([l2_s, linf_s]):
        fig = plt.figure(figsize = (4,4))
        k = 0
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                mov_avg = moving_average(norm[rank][:,2], 10)
                plt.plot(mov_avg, label=f'({j},{i})', linestyle=linestyles[k], color=colors[k], linewidth=2, 
                         marker=markers[k], markersize=5, markevery=outer_iter//5)
                k += 1
        plt.xlabel('Outer Iterations', fontsize=14)
        if n == 0:
            plt.ylabel(r'$\mathcal{E}_r^{(k)}$', fontsize=14)
        else:
            plt.ylabel(r'$\mathcal{E}_{\infty}^{(k)}$', fontsize=14)
        plt.semilogy()
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{norm_list[n]}_{var_list}.png', dpi=300)

    intfac_list = ['alpha_u', 'beta', 'gamma', 'alpha_k']
    for n, intfac_name in enumerate(intfac_list):
        fig = plt.figure(figsize = (4,4))
        k = 0
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                mov_avg = moving_average(q_s[rank][:,n+1], 10)
                plt.plot(mov_avg, label=f'({j},{i})', linestyle=linestyles[k], color=colors[k], linewidth=2, 
                             marker=markers[k], markersize=5, markevery=outer_iter//5)
                k += 1
        plt.xlabel('0uter Iterations', fontsize=14)

        if intfac_name == 'alpha_u':
            plt.ylabel(r'$\alpha^{(u)}$', fontsize=14)
            plt.legend(prop={'size': 12}, frameon=False)
        elif intfac_name == 'beta':
            plt.ylabel(r'$\beta$', fontsize=14)
        elif intfac_name == 'gamma':
            plt.ylabel(r'$\gamma$', fontsize=14)
        else:
            plt.ylabel(r'$\alpha^{(k)}$', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{intfac_name}_intfac_para.png', dpi=300)

    intfac_list = ['beta', 'gamma', 'alpha_k']
    for n, intfac_name in enumerate(intfac_list):
        fig = plt.figure(figsize = (4,4))
        k = 0
        for i in range(nx):
            for j in range(ny):
                rank = i*ny + j
                mov_avg = moving_average(q_s[rank][:,n+2]/q_s[rank][:,1], 10)
                plt.plot(mov_avg, label=f'({j},{i})', linestyle=linestyles[k], color=colors[k], linewidth=2, 
                             marker=markers[k], markersize=5, markevery=outer_iter//5)
                k += 1
        plt.xlabel('0uter Iterations', fontsize=14)

        if intfac_name == 'beta':
            plt.ylabel(r'$\beta/\alpha^{(u)}$', fontsize=14)
        elif intfac_name == 'gamma':
            plt.ylabel(r'$\gamma/\alpha^{(u)}$', fontsize=14)
        else:
            plt.ylabel(r'$\alpha^{(k)}/\alpha^{(u)}$', fontsize=14)
            plt.legend(prop={'size': 12}, frameon=False)
        plt.tight_layout()
        plt.savefig(f'pic/{methodname}_{trial}_{intfac_name}_interface_ratio.png', dpi=300)

