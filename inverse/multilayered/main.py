#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from mpi4py import MPI
from tqdm import tqdm

from losses import u_exact, k_exact, physics_loss, boundary_loss, avg_if_loss
from sample import rank_domain, fetch_interior_data, fetch_boundary_data, fetch_interface_data, fetch_uniform_mesh
from post_process import contour_prediction 


# In[2]:


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
        self.mu  = torch.nn.Parameter(kwargs["mean"],requires_grad=False)
        self.std = torch.nn.Parameter(kwargs["stdev"],requires_grad=False)
        self.k   = torch.nn.Parameter(torch.rand(1),requires_grad=True)

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


# In[3]:


def local2adjacent(cart_comm, u_int, rank, device):
    u_local = u_int.detach().cpu().numpy().copy()
    m, num_neighbors = u_local.shape
    
    left, right   = cart_comm.Shift(0, 1)
    up, down      = cart_comm.Shift(1, 1)
    
    recv_buffers = [np.empty_like(u_local[:,i]) for i in range(num_neighbors)]

    # Non-blocking send and receive operations
    req = []
    idx = 0
    for neighbor in [up, left, down, right]: # counter-clockwise
        if neighbor != MPI.PROC_NULL: 
            req.append(cart_comm.Isend(np.array(u_local[:,idx]), dest=neighbor))
            req.append(cart_comm.Irecv(recv_buffers[idx], source=neighbor))
            idx += 1

    # Wait for all non-blocking operations to complete
    MPI.Request.Waitall(req)

    u_adjacent = np.concatenate(recv_buffers, axis=0).reshape(num_neighbors, u_local.shape[0]).T
    u_adjacent_tensor = torch.tensor(u_adjacent, dtype=torch.float64).to(device)
    return u_adjacent_tensor

def exchange_interface(model, x, y, cart_comm, rank, device):
    u  = torch.zeros_like(x)
    du = torch.zeros_like(x)
    if torch.isnan(x).any():
        return u, du
    else:
        n, num_interface = x.shape
        for i in range(num_interface):
            x_col = x[:, i].reshape(-1, 1)
            y_col = y[:, i].reshape(-1, 1)

            u_col = model(x_col, y_col)

            all_x_same = torch.all(x_col.eq(x_col[0]))
            if all_x_same: #column has the same x
                du_col = model.k * torch.autograd.grad(u_col.sum(), x_col, create_graph=True)[0]
            else:
                du_col = model.k * torch.autograd.grad(u_col.sum(), y_col, create_graph=True)[0]
            u[:,i]  = u_col.flatten()
            du[:,i] = du_col.flatten()
        u_adj  = local2adjacent(cart_comm, u, rank, device)
        du_adj = local2adjacent(cart_comm, du, rank, device)
        return u_adj, du_adj

def unitify_interface_data(cart_comm, x_int, y_int, rank, device):
    m, num_neighbors = x_int.shape
    data = torch.cat((x_int, y_int), dim=0).detach().cpu().numpy()
    
    left, right   = cart_comm.Shift(0, 1)
    up, down      = cart_comm.Shift(1, 1)

    recv_buffers = [np.zeros_like(data[:,i]) for i in range(num_neighbors)]
    
    req = []
    idx = 0
    for neighbor in [up, left, down, right]: # counter-clockwise
        if neighbor != MPI.PROC_NULL:
            if neighbor > rank:
                req.append(cart_comm.Isend(np.array(data[:,idx]), dest=neighbor))
            else:
                req.append(cart_comm.Irecv(recv_buffers[idx], source=neighbor))
            idx += 1

    MPI.Request.Waitall(req)
    
    recv_data = np.concatenate(recv_buffers, axis=0).reshape(num_neighbors, m*2).T
    for i in range(num_neighbors):
        if np.sum(recv_data[:,i]) != 0:
            data[:,i] = recv_data[:,i]
    x_uni = data[:m,:] 
    y_uni = data[m:,:]
    x_uni_tensor = torch.tensor(x_uni, dtype=torch.float64).to(device)
    y_uni_tensor = torch.tensor(y_uni, dtype=torch.float64).to(device)
    return x_uni_tensor, y_uni_tensor

def gather_array(local_array, rank, size):
    total_array_shape = (size * local_array.shape[0], local_array.shape[1])

    if rank == 0:
        gathered_array = np.empty(total_array_shape, dtype=local_array.dtype)
    else:
        gathered_array = None
    comm.Gather(local_array, gathered_array, root=0)
    return gathered_array


# In[4]:

def sampling(domain, n_dom, n_bound, n_inter, rank, size, dims):
    x_dm,y_dm = fetch_interior_data(domain, n_dom)
    x_dm = x_dm.to(device)
    y_dm = y_dm.to(device)
    x_dm = x_dm.requires_grad_(True)
    y_dm = y_dm.requires_grad_(True)

    n_da = 10
    x_da = torch.full((n_da, 1), 0.5+(rank-0.5)/10)
    y_da = torch.linspace(0.2, 0.8, n_da).unsqueeze(1)
    x_da = x_da.to(device)
    y_da = y_da.to(device)
    x_da = x_da.requires_grad_(True)
    y_da = y_da.requires_grad_(True) 

    x_bc,y_bc = fetch_boundary_data(domain, n_bound, rank, size, dims)
    x_bc = x_bc.to(device)
    y_bc = y_bc.to(device)
    x_bc = x_bc.requires_grad_(True)
    y_bc = y_bc.requires_grad_(True)

    x_int0,y_int0 = fetch_interface_data(domain, n_inter, rank, size, dims)
    x_int,y_int = unitify_interface_data(cart_comm, x_int0, y_int0, rank, device)
    x_int = x_int.to(device)
    y_int = y_int.to(device)
    x_int = x_int.requires_grad_(True)
    y_int = y_int.requires_grad_(True)
    return x_dm,y_dm, x_da,y_da, x_bc,y_bc, x_int,y_int

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

def paraing(num_interface):
    Robin  = torch.ones(2, 1, device=device) # Actually [alpha, beta]
    Robin  = Robin.requires_grad_(True)
    Mu     = torch.ones(1 + 2, 1, device=device) # BC and PDE constraints Data
    Lambda = Mu * 1.
    Lambda = Lambda.requires_grad_(True)
    Bar_v  = Lambda * 0.

    optim_change  = True
    optimizer     = optim.Adam(model.parameters(), lr=5e-3)
    scheduler     = ReduceLROnPlateau(optimizer, patience=100, factor=0.95)
    #optim.LBFGS(model.parameters(),line_search_fn="strong_wolfe")
    optim_robin   = optim.Adam([Robin],maximize=True, lr=1e-4)
    
    return Robin, Lambda, Mu, Bar_v, optim_change, optimizer, scheduler, optim_robin

def intering(x_int):
    u_adj  = torch.zeros_like(x_int, device=device)
    k_adj  = torch.zeros_like(x_int, device=device)
    du_adj = torch.zeros_like(x_int, device=device)
    return u_adj, k_adj, du_adj


# In[5]:


def training(rank, epochs, model, x_dm,y_dm, x_da,y_da, x_bc,y_bc, x_int,y_int, u_adj, k_adj, du_adj, 
            Robin, Lambda, Mu, Bar_v, optimizer, scheduler, optim_robin):
    for epoch in range(epochs):
        def _closure():
            model.eval()
            pde_loss = physics_loss(model, x_dm,y_dm)
            avg_pde_loss = torch.mean(pde_loss).reshape(1, 1)
            
            bc_loss = boundary_loss(model, x_bc,y_bc)
            avg_bc_loss = torch.mean(bc_loss).reshape(1, 1)

            avg_int_loss = avg_if_loss(Robin, model,x_int,y_int, u_adj, k_adj, du_adj)
            
            data_loss     = boundary_loss(model, x_da, y_da)
            avg_data_loss = torch.mean(data_loss).reshape(1, 1)

            objective = avg_int_loss
            constr = torch.cat((avg_bc_loss, avg_data_loss, avg_pde_loss),dim=0)
            loss = objective + Lambda.T @ constr + 0.5 * Mu.T @ constr.pow(2)
            return objective, constr, loss

        def closure():
            if torch.is_grad_enabled():
                model.train()
                optimizer.zero_grad()
                optim_robin.zero_grad()
            objective, constr, loss = _closure()
            if loss.requires_grad:
                  loss.backward()
            return loss
        optimizer.step(closure)
        optim_robin.step(closure)

        objective, constr, loss = _closure()
        with torch.no_grad():
            Bar_v        = 0.99*Bar_v + 0.01*constr.pow(2)
            Mu           = 1e-2 / (torch.sqrt(Bar_v) + 1e-8)
            Lambda      += Mu * constr

        objective, constr, loss = _closure()
        #scheduler.step(loss.item())
        if epoch%20 == 0:
            print('rank %d: n = %d, objective = %.3e, constr_loss = %.3e, %.3e, %.3e'%(rank, epoch, objective, constr[0], constr[1], constr[2]))


# In[7]:


def evaluate(model, domain, test_dis, rank, size):
    model.eval()
    surrounding    = True
    x_test, y_test = fetch_uniform_mesh(domain, test_dis, surrounding)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    x_test = x_test.requires_grad_(True)
    y_test = y_test.requires_grad_(True)

    u_star     = u_exact(x_test,y_test)
    u_pred     = model(x_test,y_test).detach()
    
    x_k,y_k    = fetch_uniform_mesh(domain, [2,2], False)
    k_star     = k_exact(x_k, y_k).cpu().detach().numpy().item()
    k_pred     = model.k.cpu().detach().numpy().item()
    
    x_u_up     = torch.cat((x_test, y_test, u_star, u_pred), dim=1).cpu().detach().numpy()

    l2_u   = np.linalg.norm(x_u_up[:,2] - x_u_up[:,3]) / np.linalg.norm(x_u_up[:,2])
    linf_u = np.max(np.abs(x_u_up[:,2] - x_u_up[:,3]))

    linf_k = np.abs(k_star - k_pred)

    return l2_u.item(), linf_u.item(), linf_k.item()

def evaluate_write(model, domain, test_dis, rank, size, methodname, trial):
    model.eval()
    surrounding    = True
    x_test, y_test = fetch_uniform_mesh(domain, test_dis, surrounding)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    x_test = x_test.requires_grad_(True)
    y_test = y_test.requires_grad_(True)

    u_star     = u_exact(x_test,y_test)
    k_star     = k_exact(x_test,y_test)
    u_pred     = model(x_test,y_test).detach()

    x_u_up     = torch.cat((x_test, y_test, u_star, u_pred), dim=1).cpu().detach().numpy()
    all_x_u_up = gather_array(x_u_up, rank, size)

    pde_loss       = physics_loss(model, x_test,y_test)
    x_pde_loss     = torch.cat((x_test, y_test, pde_loss), dim=1).cpu().detach().numpy()
    all_x_pde_loss = gather_array(x_pde_loss, rank, size)

    l2_u = np.empty(1, dtype='float64')
    if rank == 0:
        filename = f'./data/{methodname}_{trial}_x_y_u_upred.dat'
        np.savetxt(filename, all_x_u_up, fmt='%.6e')

        l2_u = np.linalg.norm(all_x_u_up[:,2] - all_x_u_up[:,3]) / np.linalg.norm(all_x_u_up[:,2])
        #linf = max(abs(u_star- u_pred.numpy())).item()

        filename = f'./data/{methodname}_{trial}_x_y_pde_loss.dat'
        np.savetxt(filename, all_x_pde_loss, fmt='%.6e')
    comm.Bcast(l2_u, root=0)
    return l2_u.item()


# In[ ]:


torch.set_default_dtype(torch.float64)
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float64
print(device)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dims = [2, 1]
if size != dims[0]*dims[1]:
    raise ValueError("This setting has wrong Cartesian topology.")
cart_comm = comm.Create_cart(dims, periods=[False, False])

# Name File Sample Method
trials = 1
outer_iter = 1000
epochs     = 100
full_domain = np.array([[0., 0.],
                       [1., 1.]])

n_side = 64
# for each splitted subdomain
n_dom = n_side**2//size
n_bound = n_side// ((dims[0]+dims[1])//2)
n_inter = n_side// ((dims[0]+dims[1])//2)

test_dis = [256//dims[0], 256//dims[1]]

methodname = f'inv_multilayered_xy{dims[0]}{dims[1]}'


# In[ ]:


l2_norms = []
for trial in range(1, trials+1):
    print("*"*20 + f' run({trial}) '+"*"*20)
    domain = rank_domain(full_domain, rank, size, dims)
    x_dm,y_dm, x_da,y_da, x_bc,y_bc, x_int,y_int = sampling(domain, n_dom, n_bound, n_inter, rank, size, dims)
     
    model      = modelling(domain)
    
    num_interface = int(x_int.shape[1])
    Robin, Lambda, Mu, Bar_v, optim_change, optimizer, scheduler, optim_robin = paraing(num_interface)

    u_adj, k_adj, du_adj = intering(x_int)

    # Outer Iteration loop
    start_time = time.perf_counter()
    for count in tqdm(range(1, outer_iter + 1)):
        if count > 1 and optim_change:
            optim_change  = False
            optimizer     = torch.optim.LBFGS(model.parameters(),max_iter=7,line_search_fn='strong_wolfe')
        
        print("*"*20 + f' outer iteration ({count}) '+"*"*20) 
        training(rank, epochs, model, x_dm,y_dm, x_da,y_da, x_bc,y_bc, x_int,y_int, 
                u_adj, k_adj, du_adj, Robin, Lambda, Mu, Bar_v, optimizer, scheduler, optim_robin)
        
        u_adj, du_adj = exchange_interface(model, x_int,y_int, cart_comm, rank, device)
        
        if count % 20 == 0:
            l2_u, linf_u, linf_k = evaluate(model, domain, test_dis, rank, size)
            print('rank %d: count = %d, linf_u = %.3e, linf_k = %.3e'%(rank, count, linf_u, linf_k))

    stop_time = time.perf_counter()
    print(f'Rank: {rank}, Elapsed time: {stop_time - start_time:.2f} s')

    torch.save(model.state_dict(), f"{methodname}_{trial}_{rank}.pt")
    
    # Evaluate model
    l2_u = evaluate_write(model, domain, test_dis, rank, size, methodname, trial)
    l2_norms.append(l2_u)
    if rank == 0:
        print('total: l2_u = %.3e'%(l2_u))


# In[8]:

if rank == 0:
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


# In[13]:


### post_processing ###
if rank == 0:
    contour_prediction(test_dis, trial2[0], dims, size, methodname)
    contour_prediction(test_dis, trial2[1], dims, size, methodname)


# In[ ]:




