#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from mpi4py import MPI

from losses import physics_loss, boundary_loss, avg_if_loss
from sample import rank_domain, fetch_interior_data, fetch_boundary_data, fetch_interface_data, fetch_uniform_mesh
from post_process import contour_prediction, plot_update


# In[2]:


class ConventBlock(nn.Module):
    def __init__(self, in_N, out_N):
        super(ConventBlock, self).__init__()
        self.Ls = None
        self.net = nn.Sequential(nn.Linear(in_N, out_N, bias=True), nn.Tanh())

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
        layers.append(nn.Linear(m, out_N, bias=True))
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
    dudn = torch.zeros_like(x)
    dudt = torch.zeros_like(x)
        
    n, num_interface = x.shape
    for i in range(num_interface):
        x_col = x[:, i].reshape(-1, 1)
        y_col = y[:, i].reshape(-1, 1)

        u_col = model(x_col, y_col)
        if torch.all(x_col.eq(x_col[0])):
            dudn_col,dudt_col = torch.autograd.grad(u_col.sum(), (x_col,y_col), create_graph=True)
        else:
            dudt_col,dudn_col = torch.autograd.grad(u_col.sum(), (x_col,y_col), create_graph=True)
        u[:,i]  = u_col.flatten()
        dudn[:,i]  = dudn_col.flatten()
        dudt[:,i]  = dudt_col.flatten()
    
    u_adj  = local2adjacent(cart_comm, u, rank, device)
    dudn_adj = local2adjacent(cart_comm, dudn, rank, device)
    dudt_adj = local2adjacent(cart_comm, dudt, rank, device)
    return u_adj, dudn_adj, dudt_adj

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
    return x_dm,y_dm, x_bc,y_bc, x_int,y_int

def modelling(domain, n_h_layers, n_neurons):
    coords_stat = stats(domain)
    kwargs = {"mean":  torch.from_numpy(coords_stat[0]),
              "stdev": torch.from_numpy(coords_stat[1])}
    model = Network(in_N=2, m=n_neurons, H_Layer=n_h_layers, out_N=1, **kwargs)
    model.to(device)
    print(model)
    print(model.mu)
    print(model.std)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    model.apply(init_weights)
    return model

def paraing():
    q  = torch.ones(3, 1, device=device) # Actually [alpha, beta, gamma]
    q  = q.requires_grad_(True)
    Mu     = torch.ones(1 + 1, 1, device=device) # BC and PDE constraints
    mu_max = Mu * 1.
    Lambda = Mu * 1.
    Bar_v  = Lambda * 0.
    return q, Lambda, Mu, mu_max, Bar_v

def optimizers():
    optim_change  = False
    optimizer     = optim.LBFGS(model.parameters(),max_iter=10,line_search_fn="strong_wolfe")#optim.Adam(model.parameters())
    optim_q   = optim.Adam([q],maximize=True)
    return optim_change, optimizer, optim_q

def intering(x_int):
    u_adj  = torch.zeros_like(x_int, device=device)
    dudn_adj = torch.zeros_like(x_int, device=device)
    dudt_adj = torch.zeros_like(x_int, device=device)
    return u_adj, dudn_adj, dudt_adj

def voidlist():
    outer_s  = []
    mu_s     = []
    lambda_s = []
    constr_s = []
    object_s = []
    q_s  = []
    loss_s   = []
    l2_s  = []
    linf_s = []
    return outer_s, mu_s, lambda_s, constr_s, object_s, q_s, loss_s, l2_s, linf_s


# In[6]:

class ParaAdapt:
    def __init__(self, zeta, omega, eta, epsilon, epoch_min):
        self.zeta = zeta
        self.omega = omega
        self.eta = eta
        self.epsilon = epsilon
        self.epoch_min = epoch_min

def collect_metrics(constr, objective, loss, Lambda, Mu, q, outer_s, mu_s, lambda_s, constr_s, object_s, q_s, loss_s, count):
    outer_s.append(count)
    constr_s.append(constr.cpu().detach().numpy().flatten())
    object_s.append(objective.cpu().detach().numpy().flatten())
    q_s.append(q.cpu().detach().numpy().flatten())
    mu_s.append(Mu.cpu().numpy().flatten())
    lambda_s.append(Lambda.detach().cpu().numpy().flatten())
    loss_s.append(loss.detach().cpu().numpy().flatten())

def training(para_adapt, rank, count, epochs, model, L,
            x_dm,y_dm, x_bc,y_bc, x_int,y_int,
            u_adj, dudn_adj, dudt_adj,
            q, Lambda, Mu, mu_max, Bar_v, optimizer, optim_q,
            outer_s, mu_s, lambda_s, constr_s, object_s, q_s, loss_s):
    last_loss = torch.tensor(torch.inf).to(device)
    for epoch in range(epochs):
        def _closure():
            model.eval()
            avg_pde_loss = physics_loss(model, x_dm,y_dm, L)
            avg_bc_loss  = boundary_loss(model, x_bc,y_bc, L)
            avg_int_loss = avg_if_loss(q, model,x_int,y_int, u_adj, dudn_adj, dudt_adj)

            objective = avg_int_loss
            constr = torch.cat((avg_bc_loss, avg_pde_loss),dim=0)
            loss = objective + Lambda.T @ constr + 0.5 * Mu.T @ constr.pow(2)
            return objective, constr, loss

        def closure():
            if torch.is_grad_enabled():
                model.train()
                optimizer.zero_grad()
                optim_q.zero_grad()
            objective, constr, loss = _closure()
            if loss.requires_grad:
                  loss.backward()
            return loss

        optimizer.step(closure)
        optim_q.step(closure)
        
        objective, constr, loss = _closure()
        if epoch%10 == 0:
            print('rank %d: n = %d, loss = %.3e, objective = %.3e, constr_loss = %.3e, %.3e'%(rank, epoch, loss, objective, constr[0], constr[1]))

        with torch.no_grad():
            Bar_v        = para_adapt.zeta*Bar_v + (1-para_adapt.zeta)*constr.pow(2)
            if loss >= para_adapt.omega * last_loss or epoch == epochs-1:
                Lambda += Mu * constr
                new_mu_max = para_adapt.eta / (torch.sqrt(Bar_v)+para_adapt.epsilon)
                mu_max = torch.max(new_mu_max, mu_max)
                Mu = torch.min( torch.max( constr/(torch.sqrt(Bar_v)+para_adapt.epsilon), torch.tensor(1.) )*Mu, mu_max)
                if epoch >= para_adapt.epoch_min:
                    break
        last_loss = loss.detach()
    collect_metrics(constr, objective, loss, Lambda, Mu, q, outer_s, mu_s, lambda_s, constr_s, object_s, q_s, loss_s, count)
    return model, q, Lambda, Mu, mu_max, Bar_v, optimizer, optim_q


# In[7]:


def printing(outer_iter, epochs, outer_s, mu_s, lambda_s, constr_s, object_s,
             q_s, loss_s, l2_s, linf_s, rank):
    outer_s = np.asarray(outer_s).reshape(-1, 1)
    mu_output = np.concatenate((outer_s, np.asarray(mu_s)), axis=1)
    lambda_output = np.concatenate((outer_s, np.asarray(lambda_s)), axis=1)
    constr_output = np.concatenate((outer_s, np.asarray(constr_s)), axis=1)
    object_output = np.concatenate((outer_s, np.asarray(object_s)), axis=1)
    q_output = np.concatenate((outer_s, np.asarray(q_s)), axis=1)
    loss_output = np.concatenate((outer_s, np.asarray(loss_s)), axis=1)

    outer_iter_s = np.arange(1, outer_iter+1).reshape(-1, 1)
    l2_output   = np.concatenate((outer_iter_s, np.asarray(l2_s).reshape(-1,1)), axis=1)
    linf_output = np.concatenate((outer_iter_s, np.asarray(linf_s).reshape(-1,1)), axis=1)

    np.savetxt(f"data/{trial}_{rank}_mu.dat", mu_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_lambda.dat", lambda_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_constr.dat", constr_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_object.dat", object_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_q.dat", q_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_loss.dat", loss_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_l2.dat", l2_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_linf.dat", linf_output, fmt="%.6e", delimiter=" ")

def printing_points(x_dm,y_dm, x_bc,y_bc, x_int,y_int, trial, rank):
    data_dom = torch.cat((x_dm,y_dm), dim=1).cpu().detach().numpy()
    data_bc  = torch.cat((x_bc,y_bc), dim=1).cpu().detach().numpy()
    data_int = torch.cat((x_int.T.reshape(-1,1),y_int.T.reshape(-1,1)), dim=1).cpu().detach().numpy()
    np.savetxt(f"data/{trial}_{rank}_dom.dat", data_dom, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_bc.dat", data_bc, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_int.dat", data_int, fmt="%.6e", delimiter=" ")

def evaluate_write(model, test_dis, dims, rank, size, methodname, trial, write=False):
    model.eval()
    x_y_u_exact = np.loadtxt(f'./ref_data/helm_l{int(L.cpu())}_xy{dims[0]}{dims[1]}_mesh{test_dis[0]}_u_ref_rank{rank}.dat')
    x_test = torch.from_numpy(x_y_u_exact[:,0].reshape(-1,1)).to(device)
    y_test = torch.from_numpy(x_y_u_exact[:,1].reshape(-1,1)).to(device)

    u_pred     = model(x_test,y_test).cpu().detach().numpy()
    x_y_u_up = np.concatenate((x_y_u_exact, u_pred), axis=1)
    if write == False:
        l2 = np.linalg.norm(x_y_u_up[:,-2] - x_y_u_up[:,-1]) / np.linalg.norm(x_y_u_up[:,-2])
        linf = max(abs(x_y_u_up[:,-2] - x_y_u_up[:,-1]))
        return l2.item(), linf.item()
    else:
        all_x_y_u_up = gather_array(x_y_u_up, rank, size)
        l2 = np.empty(1, dtype='float64')
        linf = np.empty(1, dtype='float64')
        if rank == 0:
            filename = f'./data/{methodname}_{trial}_x_y_u_upred.dat'
            np.savetxt(filename, all_x_y_u_up, fmt='%.6e')
            # Global relative l2 norm
            l2 = np.linalg.norm(all_x_y_u_up[:,-2] - all_x_y_u_up[:,-1]) / np.linalg.norm(all_x_y_u_up[:,-2])
            linf = max(abs(all_x_y_u_up[:,-2] - all_x_y_u_up[:,-1]))
        comm.Bcast(l2,root=0)
        comm.Bcast(linf,root=0)
        return l2.item(), linf.item()


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

dims = [5, 5]
if size != dims[0]*dims[1]:
    raise ValueError("This setting has wrong Cartesian topology.")
cart_comm = comm.Create_cart(dims, periods=[False, False])

# Name File Sample Method
trials = 1
outer_iter = 10
epochs     = 100
L      = torch.tensor([5.], device = device)
full_domain = np.array([[0., 0.],
                       [1., 1.]])

mesh = 160
full_dom_dis = [mesh, mesh]
# for each splitted subdomain
n_dom = mesh**2//size
n_bound = int( mesh// ((dims[0]+dims[1])/2) )
n_inter = int( mesh// ((dims[0]+dims[1])/2) ) 

test_dis = [360//dims[0] ,360//dims[1]]

n_h_layers = 3
n_neurons  = 20

para_adapt = ParaAdapt(zeta=0.999, omega=0.999, eta=torch.tensor([[10.],[0.01]]), epsilon=1e-16, epoch_min=50)
methodname = f'helm_multi_l{int(L.cpu())}_xy{dims[0]}{dims[1]}_nn{n_h_layers}_{n_neurons}_dom_dis{mesh}'


# In[ ]:


l2_norms = []
for trial in range(1, trials+1):
    print("*"*20 + f' run({trial}) '+"*"*20)
    domain = rank_domain(full_domain, rank, size, dims)
    x_dm,y_dm, x_bc,y_bc, x_int,y_int = sampling(domain, n_dom, n_bound, n_inter, rank, size, dims)
    printing_points(x_dm,y_dm, x_bc,y_bc, x_int,y_int, trial, rank)
     
    model      = modelling(domain, n_h_layers, n_neurons)
    
    q, Lambda, Mu, mu_max, Bar_v = paraing()
    optim_change, optimizer, optim_q = optimizers()

    u_adj, dudn_adj, dudt_adj = intering(x_int)

    outer_s, mu_s, lambda_s, constr_s, object_s, q_s, loss_s, l2_s, linf_s = voidlist()

    # Training loop
    start_time = time.perf_counter()
    for count in range(1, outer_iter + 1):
        print("*"*20 + f' outer iteration ({count}) '+"*"*20) 
        model, q, Lambda, Mu, mu_max, Bar_v, optimizer, optim_q = training(para_adapt, rank, count, epochs, model, L,
                x_dm,y_dm, x_bc,y_bc, x_int,y_int,
                u_adj, dudn_adj, dudt_adj,
                q, Lambda, Mu, mu_max, Bar_v, optimizer, optim_q,
                outer_s, mu_s, lambda_s, constr_s, object_s, q_s, loss_s)
        
        u_adj, dudn_adj, dudt_adj = exchange_interface(model, x_int,y_int, cart_comm, rank, device)

        l2, linf = evaluate_write(model, test_dis, dims, rank, size, methodname, trial)
        l2_s.append(l2)
        linf_s.append(linf)

        if count % 10==0:
            # Global relative l2 norm
            l2, linf = evaluate_write(model, test_dis, dims, rank, size, methodname, trial, write=True)
            if rank == 0:
                print('l2 norm = %.3e, linf norm = %.3e'%(l2, linf))
            printing(count, epochs, outer_s, mu_s, lambda_s, constr_s, object_s,
                     q_s, loss_s, l2_s, linf_s, rank)            
            torch.save(model.state_dict(), f"{methodname}_{trial}_{rank}.pt")
    stop_time = time.perf_counter()
    print(f'Rank: {rank}, Elapsed time: {stop_time - start_time:.2f} s')

    # Evaluate model
    l2, linf = evaluate_write(model, test_dis, dims, rank, size, methodname, trial, write=True)
    l2_norms.append(l2)


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

    data_summary = [np.mean(l2_norms), np.std(l2_norms), trial2[1], l2_norms[trial2[1]-1],
                                                     trial2[0], l2_norms[trial2[0]-1]]
    data_summary = np.asarray(data_summary)
    filename = f'./data/{methodname}_summary.dat'
    np.savetxt(filename, data_summary, fmt='%.6e')


# In[13]:


### post_processing ###
if rank == 0:
    contour_prediction(test_dis, trial2[1], dims, size, methodname)
    plot_update(trial2[1], dims, size, outer_iter, methodname)


# In[ ]:




