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




