#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm

from losses import u_exact, physics_loss, boundary_loss, avg_if_loss, s_exact
from sample import rank_domain, fetch_interior_data, fetch_boundary_data, fetch_interface_data, fetch_uniform_mesh
from post_process import contour_prediction, plot_exchange, plot_update, sample_residual_distribution


# In[2]:


def sampling(full_domain, domain_0, domain_1, n_dom, n_bound, n_inter):
    x_dm,y_dm = fetch_interior_data(full_domain, n_dom)
    x_dm = x_dm.to(device)
    y_dm = y_dm.to(device)
    
    mask = (x_dm > 0.25) & (x_dm < 0.75) & (y_dm > 0.25) & (y_dm < 0.75)
    x_dm_1 = x_dm[mask][:,None]
    y_dm_1 = y_dm[mask][:,None]
    x_dm_0 = x_dm[~mask][:,None]
    y_dm_0 = y_dm[~mask][:,None]
    x_dm_1 = x_dm_1.requires_grad_(True)
    y_dm_1 = y_dm_1.requires_grad_(True)
    x_dm_0 = x_dm_0.requires_grad_(True)
    y_dm_0 = y_dm_0.requires_grad_(True)

    x_bc,y_bc = fetch_boundary_data(domain_0, n_bound)
    x_bc = x_bc.to(device)
    y_bc = y_bc.to(device)
    x_bc = x_bc.requires_grad_(True)
    y_bc = y_bc.requires_grad_(True)

    x_bc_1 = torch.full([1,1], float('nan'))
    y_bc_1 = torch.full([1,1], float('nan'))
    x_bc_1 = x_bc_1.to(device)
    y_bc_1 = y_bc_1.to(device)
    
    x_int,y_int = fetch_boundary_data(domain_1, n_bound)
    x_int = x_int.to(device)
    y_int = y_int.to(device)
    x_int = x_int.requires_grad_(True)
    y_int = y_int.requires_grad_(True)
    return x_dm_0,y_dm_0,x_dm_1,y_dm_1, x_bc,y_bc, x_bc_1,y_bc_1, x_int,y_int


# In[3]:


class ConventBlock(nn.Module):
    def __init__(self, in_N, out_N):
        super(ConventBlock, self).__init__()
        self.Ls = None
        self.net = nn.Sequential(nn.Linear(in_N, out_N), nn.Tanh())

    def forward(self, x):
        out = self.net(x)
        return out

class Network(torch.nn.Module):
    def __init__(self, in_N, m, H_Layer, out_N, **kwargs):
        super(Network, self).__init__()
        self.mu = torch.nn.Parameter(kwargs["mean"],requires_grad=False)
        self.std = torch.nn.Parameter(kwargs["stdev"],requires_grad=False)
        layers = []
        layers.append(ConventBlock(in_N, m))
        for i in range(0, H_Layer-1):
            layers.append(ConventBlock(m, m))
         # output layer
        layers.append(nn.Linear(m, out_N))
        # total layers
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        data = torch.cat((x,y), dim=1)
        # normalize the input
        data = (data - self.mu)/self.std
        out = self.net(data)
        return out

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)
                
def stats(domain_coords):
    dim      = len(domain_coords)
    coords_mean = (domain_coords[1] + domain_coords[0]) / 2
    coords_std  = (domain_coords[1] - domain_coords[0]) / np.sqrt(12)
    return np.vstack((coords_mean, coords_std))

def modelling(domain):
    coords_stat = stats(domain)
    kwargs = {"mean":  torch.from_numpy(coords_stat[0]),
              "stdev": torch.from_numpy(coords_stat[1])}
    model = Network(in_N=2, m=20, H_Layer=3, out_N=1, **kwargs)
    model.to(device)
    print(model)
    print(model.mu)
    print(model.std)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    model.apply(init_weights)
    return model


# In[4]:


def paraing(n_lambda, model):
    Alpha  = torch.ones(2, 1, device=device) # Actually [alpha, beta]
    Alpha  = Alpha.requires_grad_(True)
    Mu     = torch.ones(n_lambda, 1, device=device) # BC, PDE constraints
    Lambda = Mu * 1.
    Bar_v  = Lambda * 0.

    optim_change  = True
    optimizer     = optim.Adam(model.parameters())#optim.LBFGS(model.parameters(),lr=1e-1,line_search_fn="strong_wolfe")
    optim_alpha   = optim.Adam([Alpha],maximize=True, lr=1e-3)

    return Alpha, Lambda, Mu, Bar_v, optim_change, optimizer, optim_alpha

def intering(x_int):
    u_adj  = torch.zeros_like(x_int, device=device)
    du_adj = torch.zeros_like(x_int, device=device)
    return u_adj, du_adj

def voidlist():
    mu_s     = []
    lambda_s = []
    constr_s = []
    object_s = []
    alpha_s  = []
    f_g_s    = []
    l2_s  = []
    linf_s = []
    return mu_s, lambda_s, constr_s, object_s, alpha_s, f_g_s, l2_s, linf_s


# In[5]:


def exchange_interface(model, x, y, rank):
    u    = model(x, y)
    du   = torch.zeros_like(x).to(device)
    
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)

    mask = y == 0.25 # Bottom edge (x, 0.25)
    du[mask] = u_y[mask]
    
    mask = x == 0.75 # Right edge (0.75, y)
    du[mask] = u_x[mask]
    
    mask = y == 0.75 # Top edge (x, 0.75)
    du[mask] = u_y[mask]
    
    mask = x == 0.25 # Left edge (0.25, y)
    du[mask] = u_x[mask]
    
    u    = u.detach()
    du   = du.detach()
    return u, du


# In[6]:


def printing_points(x_dm_0,y_dm_0,x_dm_1,y_dm_1, x_bc,y_bc, x_int,y_int, trial):
    data_dom_0 = torch.cat((x_dm_0,y_dm_0), dim=1).cpu().detach().numpy()
    data_dom_1 = torch.cat((x_dm_1,y_dm_1), dim=1).cpu().detach().numpy()
    data_bc  = torch.cat((x_bc,y_bc), dim=1).cpu().detach().numpy()
    data_int = torch.cat((x_int.T.reshape(-1,1),y_int.T.reshape(-1,1)), dim=1).cpu().detach().numpy()
    np.savetxt(f"data/{trial}_dom_0.dat", data_dom_0, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_dom_1.dat", data_dom_1, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_bc.dat", data_bc, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_int.dat", data_int, fmt="%.6e", delimiter=" ")

def printing(outer_iter, epochs, mu_s, lambda_s, constr_s, object_s, 
             alpha_s, f_g_s, l2_s, linf_s, rank):
    outer_iter_s = np.arange(1, outer_iter+1).reshape(-1, 1)
    mu_output = np.concatenate((outer_iter_s, np.asarray(mu_s)), axis=1)
    lambda_output = np.concatenate((outer_iter_s, np.asarray(lambda_s)), axis=1)
    constr_output = np.concatenate((outer_iter_s, np.asarray(constr_s)), axis=1)
    object_output = np.concatenate((outer_iter_s, np.asarray(object_s)), axis=1)
    alpha_output = np.concatenate((outer_iter_s, np.asarray(alpha_s)), axis=1)
    f_g_s_output = np.concatenate((outer_iter_s, np.asarray(f_g_s)), axis=1)
    l2_output = np.concatenate((outer_iter_s, np.asarray(l2_s)[:,None]), axis=1)
    linf_output = np.concatenate((outer_iter_s, np.asarray(linf_s)[:,None]), axis=1)

    np.savetxt(f"data/{trial}_{rank}_mu.dat", mu_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_lambda.dat", lambda_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_constr.dat", constr_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_object.dat", object_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_alpha.dat", alpha_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_f_g_loss.dat", f_g_s_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_l2.dat", l2_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_linf.dat", linf_output, fmt="%.6e", delimiter=" ")


# In[7]:


def training(rank, epochs, model, x_dm,y_dm, x_bc,y_bc, x_int,y_int, 
             u_adj, du_adj, Alpha, Lambda, Mu, Bar_v, optimizer, optim_alpha,
             constr_s, object_s, alpha_s, f_g_s, mu_s, lambda_s):
    for epoch in range(epochs):
        def _closure():
            model.eval()
            pde_loss = physics_loss(model, x_dm,y_dm)
            avg_pde_loss = torch.mean(pde_loss).reshape(1, 1)

            bc_loss = boundary_loss(model, x_bc,y_bc)
            avg_bc_loss = torch.mean(bc_loss).reshape(1, 1)

            f_loss, g_loss, r_loss, avg_int_loss = avg_if_loss(Alpha, model, x_int,y_int, u_adj, du_adj)

            objective = avg_int_loss
            constr = torch.cat((avg_bc_loss, avg_pde_loss),dim=0)
            loss = objective + Lambda.T @ constr + 0.5 * Mu.T @ constr.pow(2)
            return objective, constr, loss, torch.cat((f_loss, g_loss, r_loss),dim=0)

        def closure():
            if torch.is_grad_enabled():
                model.train()
                optimizer.zero_grad()
                optim_alpha.zero_grad()
            objective, constr, loss, f_g_loss = _closure()
            if loss.requires_grad:
                  loss.backward()
            return loss
        optimizer.step(closure)
        optim_alpha.step(closure)

        objective, constr, loss, f_g_loss = _closure()
        with torch.no_grad():
            Bar_v        = 0.99*Bar_v + 0.01*constr.pow(2)
            Mu           = 1e-2 / (torch.sqrt(Bar_v) + 1e-8)
            Lambda      += Mu * constr
            
        if epoch%5 == 0:
            print('rank %d: n = %d, objective = %.3e, constr_loss = %.3e, %.3e'%(rank, epoch, objective,
            constr[0], constr[1]))

    constr_s.append(constr.cpu().detach().numpy().flatten())
    object_s.append(objective.cpu().detach().numpy().flatten())
    alpha_s.append(Alpha.cpu().detach().numpy().flatten()) #record Alpha = [alpha, beta]
    f_g_s.append(f_g_loss.cpu().detach().numpy().flatten())
    mu_s.append(Mu.cpu().numpy().flatten())
    lambda_s.append(Lambda.detach().cpu().numpy().flatten())


# In[8]:


def evaluate_write(model_0, model_1, full_domain, test_dis, write, methodname, trial):
    model_0.eval()
    model_1.eval()
    surrounding    = True
    x_test, y_test = fetch_uniform_mesh(full_domain, test_dis, surrounding)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    mask = (x_test > 0.25) & (x_test < 0.75) & (y_test > 0.25) & (y_test < 0.75)
    x_test_1 = x_test[mask][:,None]
    y_test_1 = y_test[mask][:,None]
    x_test_0 = x_test[~mask][:,None]
    y_test_0 = y_test[~mask][:,None]

    u_star_0   = u_exact(x_test_0,y_test_0).cpu().detach().numpy()
    u_pred_0   = model_0(x_test_0,y_test_0).cpu().detach().numpy()
    l2_0 = np.linalg.norm(u_star_0 - u_pred_0) / np.linalg.norm(u_star_0)
    linf_0 = np.max(np.abs(u_star_0 - u_pred_0))
    
    u_star_1   = u_exact(x_test_1,y_test_1).cpu().detach().numpy()
    u_pred_1   = model_1(x_test_1,y_test_1).cpu().detach().numpy()
    l2_1 = np.linalg.norm(u_star_1 - u_pred_1) / np.linalg.norm(u_star_1)
    linf_1 = np.max(np.abs(u_star_1 - u_pred_1))
    
    u_star   = u_exact(x_test,y_test).cpu().detach().numpy()
    u_pred   = np.zeros_like(u_star)
    u_pred[~mask]= u_pred_0.flatten()
    u_pred[mask] = u_pred_1.flatten()
    if write==True:
        x_u_up     = np.concatenate((x_test, y_test, u_star, u_pred), axis=1)
        #l2   = np.linalg.norm(x_u_up[:,-2] - x_u_up[:,-1]) / np.linalg.norm(x_u_up[:,-2])
        #linf = np.max(np.abs(x_u_up[:,-2] - x_u_up[:,-1]))
        np.savetxt(f'./data/{methodname}_{trial}_x_y_u_upred.dat', x_u_up, fmt='%.6e')
    return l2_0.item(), linf_0.item(), l2_1.item(), linf_1.item()


def evaluate(model_0, model_1, full_domain, n_test):
    model_0.eval()
    model_1.eval()
    x_test, y_test = fetch_interior_data(full_domain, n_test)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    mask = (x_test > 0.25) & (x_test < 0.75) & (y_test > 0.25) & (y_test < 0.75)
    x_test_1 = x_test[mask][:,None]
    y_test_1 = y_test[mask][:,None]
    x_test_0 = x_test[~mask][:,None]
    y_test_0 = y_test[~mask][:,None]

    u_star_0   = u_exact(x_test_0,y_test_0)
    u_pred_0   = model_0(x_test_0,y_test_0).detach()
    x_u_up_0   = torch.cat((x_test_0, y_test_0, u_star_0, u_pred_0), dim=1).cpu().detach().numpy()
    u_star_1   = u_exact(x_test_1,y_test_1)
    u_pred_1   = model_1(x_test_1,y_test_1).detach()
    x_u_up_1   = torch.cat((x_test_1, y_test_1, u_star_1, u_pred_1), dim=1).cpu().detach().numpy()
        
    x_u_up     = np.concatenate((x_u_up_0, x_u_up_1), axis=0)
    l2   = np.linalg.norm(x_u_up[:,-2] - x_u_up[:,-1]) / np.linalg.norm(x_u_up[:,-2])
    linf = np.max(np.abs(x_u_up[:,-2] - x_u_up[:,-1]))
    return l2


# In[9]:


torch.set_default_dtype(torch.float64)
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float64
print(device)

# Name File Sample Method
trials = 5
outer_iter = 1000
epochs     = 100

full_domain = np.array([[0., 0.],
                       [1., 1.]])
domain_0 = full_domain
domain_1 = np.array([[0.25, 0.25],
                    [0.75, 0.75]])
                    
# for the global domain
n_dom = 400
n_bound = 80
n_inter = 80

n_test = 1002001
test_dis = [100, 100]
methodname = f'psn_xpinn_dom{n_dom}_bd{n_bound}'


# In[10]:


l2_norms = []
for trial in range(1, trials+1):
    print("*"*20 + f' run({trial}) '+"*"*20)
    x_dm_0,y_dm_0,x_dm_1,y_dm_1, x_bc_0,y_bc_0, x_bc_1,y_bc_1, x_int,y_int = sampling(full_domain, domain_0, domain_1, n_dom, n_bound, n_inter)
    printing_points(x_dm_0,y_dm_0,x_dm_1,y_dm_1, x_bc_0,y_bc_0, x_int,y_int, trial)

    model_0    = modelling(domain_0)
    model_1    = modelling(domain_1)

    n_lambda = 2
    Alpha_0, Lambda_0, Mu_0, Bar_v_0, optim_change, optimizer_0, optim_alpha_0 = paraing(n_lambda, model_0)
    Alpha_1, Lambda_1, Mu_1, Bar_v_1, optim_change, optimizer_1, optim_alpha_1 = paraing(n_lambda, model_1)

    u_adj_0, du_adj_0 = intering(x_int)
    u_adj_1, du_adj_1 = intering(x_int)
    
    mu_s_0, lambda_s_0, constr_s_0, object_s_0, alpha_s_0, f_g_s_0, l2_s_0, linf_s_0 = voidlist()
    mu_s_1, lambda_s_1, constr_s_1, object_s_1, alpha_s_1, f_g_s_1, l2_s_1, linf_s_1 = voidlist()
    
    # Training loop
    for count in range(1, outer_iter + 1):
        print("*"*20 + f' outer iteration ({count}) '+"*"*20)
        if count > 1 and optim_change:
            optim_change  = False
            optimizer_0 = optim.LBFGS(model_0.parameters(),lr=1e-1,line_search_fn="strong_wolfe")
            optimizer_1 = optim.LBFGS(model_1.parameters(),lr=1e-1,line_search_fn="strong_wolfe")
            
        rank = 0
        training(rank, epochs, model_0, x_dm_0,y_dm_0, x_bc_0,y_bc_0, x_int,y_int, 
             u_adj_0, du_adj_0, Alpha_0, Lambda_0, Mu_0, Bar_v_0, optimizer_0, optim_alpha_0,
             constr_s_0, object_s_0, alpha_s_0, f_g_s_0, mu_s_0, lambda_s_0)
        rank = 1
        training(rank, epochs, model_1, x_dm_1,y_dm_1, x_bc_1,y_bc_1, x_int,y_int, 
             u_adj_1, du_adj_1, Alpha_1, Lambda_1, Mu_1, Bar_v_1, optimizer_1, optim_alpha_1,
             constr_s_1, object_s_1, alpha_s_1, f_g_s_1, mu_s_1, lambda_s_1)
        
        rank = 0
        u_adj_1, du_adj_1 = exchange_interface(model_0, x_int, y_int, rank)
        rank = 1
        u_adj_0, du_adj_0 = exchange_interface(model_1, x_int, y_int, rank)
        
        l2_0, linf_0, l2_1, linf_1 = evaluate_write(model_0, model_1, full_domain, test_dis, False, methodname, trial)
        l2_s_0.append(l2_0)
        linf_s_0.append(linf_0)
        l2_s_1.append(l2_1)
        linf_s_1.append(linf_1)
        if count % 20 == 0:
            rank = 0
            print('rank %d: count = %d, l2 norm = %.3e, linf norm = %.3e'%(rank, count, l2_0, linf_0))
            printing(count, epochs, mu_s_0, lambda_s_0, constr_s_0, object_s_0, alpha_s_0, f_g_s_0, l2_s_0, linf_s_0, rank)
            rank = 1
            print('rank %d: count = %d, l2 norm = %.3e, linf norm = %.3e'%(rank, count, l2_1, linf_1))
            printing(count, epochs, mu_s_1, lambda_s_1, constr_s_1, object_s_1, alpha_s_1, f_g_s_1, l2_s_1, linf_s_1, rank)
    
    torch.save(model_0.state_dict(), f"{methodname}_{trial}_0.pt")
    torch.save(model_1.state_dict(), f"{methodname}_{trial}_1.pt")
    
    rank = 0
    printing(outer_iter, epochs, mu_s_0, lambda_s_0, constr_s_0, object_s_0, alpha_s_0, f_g_s_0, l2_s_0, linf_s_0, rank)
    rank = 1
    printing(outer_iter, epochs, mu_s_1, lambda_s_1, constr_s_1, object_s_1, alpha_s_1, f_g_s_1, l2_s_1, linf_s_1, rank)
    
    # Evaluate
    l2 = evaluate(model_0, model_1, full_domain, n_test)
    l2_0, linf_0, l2_1, linf_1 = evaluate_write(model_0, model_1, full_domain, test_dis, True, methodname, trial)
    l2_norms.append(l2)
    print('l2 norm = %.3e'%(l2))
    


# In[ ]:


print('mean L_2: %2.3e' % np.mean(l2_norms))
print('std  L_2: %2.3e' % np.std(l2_norms))
print('*'*20)
print('total trials: ', trials)

trial2 = np.array([l2_norms.index(max(l2_norms)),
                  l2_norms.index(min(l2_norms))]) + 1
print('worst trial: ', trial2[0])
print(f"relative l2 error :{l2_norms[trial2[0]-1]:2.3e}")
print('best trial: ', trial2[1])
print(f"relative l2 error :{l2_norms[trial2[1]-1]:2.3e}")

data_summary = [np.mean(l2_norms), np.std(l2_norms), trial2[1], l2_norms[trial2[1]-1],
                                                 trial2[0], l2_norms[trial2[0]-1]]
data_summary = np.asarray(data_summary)
filename = f'./data/{methodname}_summary.dat'
np.savetxt(filename, data_summary, fmt='%.6e')


# In[ ]:


### post_processing ###
contour_prediction(test_dis, trial2[0], methodname)
contour_prediction(test_dis, trial2[1], methodname)

#sample_residual_distribution(omega, test_dis, trial2[0], dims, size, methodname)
#sample_residual_distribution(omega, test_dis, trial2[1], dims, size, methodname)

#plot_update(trial2[0], outer_iter, epochs, methodname)
#plot_update(trial2[1], outer_iter, epochs, methodname)


# In[ ]:



