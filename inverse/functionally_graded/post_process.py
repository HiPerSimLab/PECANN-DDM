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


def boundary_location(theta):
    rho       = 1 + np.cos(theta)*np.sin(4*theta)
    radius    = rho
    x         = 6*0.55*radius*np.cos(theta)
    y         = 6*0.75*radius*np.sin(theta)
    return x,y

def contour_prediction(n_test, trial, dims, size, methodname):    
    u_up = np.loadtxt(f'./data/{methodname}_{trial}_x_y_u_upred.dat')
    
    x      = u_up[:,0]
    y      = u_up[:,1]
    u_star = u_up[:,2]
    u_pred = u_up[:,3]
    u_err  = abs(u_star - u_pred)
    k_star = u_up[:,4]
    k_pred = u_up[:,5]
    k_err  = abs(k_star - k_pred)
    
    varlist = ['u_star', 'u_pred', 'u_err', 'k_star', 'k_pred', 'k_err']
    for k, var in enumerate([u_star, u_pred, u_err, k_star, k_pred, k_err]):
        fig = plt.figure(figsize = (6,5)) #(x,y): 1D: (9,5); 2D: (6,5)
        gs = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.4)

        vmax = np.max(var)
        vmin = np.min(var)

        ax = plt.subplot(gs[0,0])
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
        plt.close()


