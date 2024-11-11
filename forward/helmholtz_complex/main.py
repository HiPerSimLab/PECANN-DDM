#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from mpi4py import MPI
from tqdm import tqdm

from losses import physics_loss, boundary_loss, interface_loss, u_exact
from sample import rank_domain, fetch_interior_data, fetch_boundary_data, fetch_interface_data, fetch_uniform_mesh
from post_process import contour_prediction, plot_update


# In[3]:


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


# In[21]:


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

def modelling(domain, n_h_layers, n_neurons, n_output):
    coords_stat = stats(domain)
    kwargs = {"mean":  torch.from_numpy(coords_stat[0]),
              "stdev": torch.from_numpy(coords_stat[1])}
    model = Network(in_N=2, m=n_neurons, H_Layer=n_h_layers, out_N=n_output, **kwargs)
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
    Mu_max = Mu * 1.
    Lambda = Mu * 1.
    Bar_v  = Lambda * 0.
    return q, Lambda, Mu, Mu_max, Bar_v

def optimizers():
    optim_q   = optim.Adam([q],maximize=True)
    optim_change  = False
    optimizer     = optim.LBFGS(model.parameters(),line_search_fn="strong_wolfe")#optim.Adam(model.parameters())#
    return optim_change, optimizer, optim_q

def intering(x_int):
    u_adj    = []
    dudn_adj = []
    dudt_adj = []
    n, num_interface = x_int.shape
    for i in range(num_interface):
        u_adj.append( torch.zeros((n, 2), device=device) )
        dudn_adj.append( torch.zeros((n, 2), device=device) )
        dudt_adj.append( torch.zeros((n, 2), device=device) )
    return u_adj, dudn_adj, dudt_adj

def voidlist():
    outer_s  = []
    mu_s     = []
    lambda_s = []
    constr_s = []
    object_s = []
    q_s      = []
    loss_s   = []
    l2_s  = []
    linf_s = []
    return outer_s, mu_s, lambda_s, constr_s, object_s, q_s, loss_s, l2_s, linf_s


# In[ ]:


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

def local2adjacent(cart_comm, u_local, rank, device):
    left, right   = cart_comm.Shift(0, 1)
    up, down      = cart_comm.Shift(1, 1)
    
    num_neighbors = len(u_local)
    recv_buffers = [np.empty_like(u_local[i]) for i in range(num_neighbors)]

    # Non-blocking send and receive operations
    req = []
    idx = 0
    for neighbor in [up, left, down, right]: # counter-clockwise
        if neighbor != MPI.PROC_NULL:
            req.append(cart_comm.Isend(u_local[idx], dest=neighbor))
            req.append(cart_comm.Irecv(recv_buffers[idx], source=neighbor))
            idx += 1

    # Wait for all non-blocking operations to complete
    MPI.Request.Waitall(req)

    u_adjacent_tensor = [torch.tensor(recv_buffers[i], dtype=torch.float64).to(device) for i in range(num_neighbors)]
    return u_adjacent_tensor

def exchange_interface(model, x,y, cart_comm, rank, device):
    u_local    = []
    dudn_local = []
    dudt_local = []
    n, num_interface = x.shape
    for i in range(num_interface):
        x_side = x[:, i:i+1]
        y_side = y[:, i:i+1]
        u_side = model(x_side, y_side)
        u_side_re, u_side_im = u_side[:,0:1], u_side[:,1:2]

        all_x_same = torch.all(x_side.eq(x_side[0]))
        if all_x_same: #column has the same x
            dudn_side_re, dudt_side_re = torch.autograd.grad(u_side_re.sum(),(x_side, y_side), create_graph=True,retain_graph=True)
            dudn_side_im, dudt_side_im = torch.autograd.grad(u_side_im.sum(),(x_side, y_side), create_graph=True,retain_graph=True)
        else:
            dudt_side_re, dudn_side_re = torch.autograd.grad(u_side_re.sum(),(x_side, y_side), create_graph=True,retain_graph=True)
            dudt_side_im, dudn_side_im = torch.autograd.grad(u_side_im.sum(),(x_side, y_side), create_graph=True,retain_graph=True)

        dudn_side    = torch.cat((dudn_side_re, dudn_side_im), dim=1)
        dudt_side    = torch.cat((dudt_side_re, dudt_side_im), dim=1)

        u_local.append(u_side.detach().cpu().numpy())
        dudn_local.append(dudn_side.detach().cpu().numpy())
        dudt_local.append(dudt_side.detach().cpu().numpy())

    u_adj = local2adjacent(cart_comm, u_local, rank, device)
    dudn_adj = local2adjacent(cart_comm, dudn_local, rank, device)
    dudt_adj = local2adjacent(cart_comm, dudt_local, rank, device)
    return u_adj, dudn_adj, dudt_adj

def gather_array(local_array, rank, size):
    total_array_shape = (size * local_array.shape[0], local_array.shape[1])

    if rank == 0:
        gathered_array = np.empty(total_array_shape, dtype=local_array.dtype)
    else:
        gathered_array = None
    comm.Gather(local_array, gathered_array, root=0)
    return gathered_array


# In[18]:


class ParaAdapt:
    def __init__(self, zeta, omega, eta, epsilon, epoch_min):
        self.zeta = zeta
        self.omega = omega
        self.eta = eta
        self.epsilon = epsilon
        self.epoch_min = epoch_min

def collect_metrics(count, Mu, Lambda, constr, objective, loss, q, 
                    outer_s, mu_s, lambda_s, constr_s, object_s, loss_s, q_s):
    outer_s.append(count)
    constr_s.append(constr.cpu().detach().numpy().flatten())
    object_s.append(objective.cpu().detach().numpy().flatten())
    mu_s.append(Mu.cpu().numpy().flatten())
    lambda_s.append(Lambda.detach().cpu().numpy().flatten())
    loss_s.append(loss.detach().cpu().numpy().flatten())
    q_s.append(q.detach().cpu().numpy().flatten())


# In[27]:


def printing(trial, outer_s, mu_s, lambda_s, constr_s, object_s, loss_s, q_s, l2_s, linf_s, rank):
    outer_s = np.asarray(outer_s).reshape(-1, 1)
    mu_output = np.concatenate((outer_s, np.asarray(mu_s)), axis=1)
    lambda_output = np.concatenate((outer_s, np.asarray(lambda_s)), axis=1)
    constr_output = np.concatenate((outer_s, np.asarray(constr_s)), axis=1)
    object_output = np.concatenate((outer_s, np.asarray(object_s)), axis=1)
    loss_output = np.concatenate((outer_s, np.asarray(loss_s)), axis=1)
    q_output = np.concatenate((outer_s, np.asarray(q_s)), axis=1)

    outer_iters = np.arange(1, outer_s[-1]+1)[:,None]
    l2_output   = np.concatenate((outer_iters, np.asarray(l2_s)), axis=1)
    linf_output = np.concatenate((outer_iters, np.asarray(linf_s)), axis=1)

    np.savetxt(f"data/{trial}_{rank}_mu.dat", mu_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_lambda.dat", lambda_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_constr.dat", constr_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_object.dat", object_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_loss.dat", loss_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_q.dat", q_output, fmt="%.6e", delimiter=" ")

    np.savetxt(f"data/{trial}_{rank}_l2.dat", l2_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_linf.dat", linf_output, fmt="%.6e", delimiter=" ")

def printing_points(x_dm,y_dm, x_bc,y_bc, x_int,y_int, trial, rank):
    data_dom = torch.cat((x_dm,y_dm), dim=1).cpu().detach().numpy()
    data_bc  = torch.cat((x_bc,y_bc), dim=1).cpu().detach().numpy()
    data_int = torch.cat((x_int.T.reshape(-1,1),y_int.T.reshape(-1,1)), dim=1).cpu().detach().numpy()
    np.savetxt(f"data/{trial}_{rank}_dom.dat", data_dom, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_bc.dat", data_bc, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_{rank}_int.dat", data_int, fmt="%.6e", delimiter=" ")

def evaluate_write(model, domain, test_dis, methodname, trial, write=False):
    model.eval()
    x_test, y_test = fetch_uniform_mesh(domain, test_dis, surrounding=True)
    x_test         = x_test.to(device).requires_grad_(True)
    y_test         = y_test.to(device).requires_grad_(True)
    
    u_star     = u_exact(x_test, y_test, k, theta).cpu().detach().numpy()
    u_pred     = model(x_test, y_test).cpu().detach().numpy()

    if write == False:
        l2   = np.linalg.norm(u_star - u_pred, axis=0) / np.linalg.norm(u_star, axis=0)
        linf = np.max(np.abs(u_star - u_pred), axis=0)
        return l2, linf
    else:
        x_test   = x_test.cpu().detach().numpy()
        y_test   = y_test.cpu().detach().numpy()
        xy_u_up  = np.concatenate((x_test, y_test, u_star, u_pred), axis=1)
        
        all_xy_u_up = gather_array(xy_u_up, rank, size)
        l2 = np.empty(2, dtype='float64')
        linf = np.empty(2, dtype='float64')
        if rank == 0:
            filename = f'./data/{methodname}_{trial}_xy_u_upred.dat'
            np.savetxt(filename, all_xy_u_up, fmt='%.6e')
            # Global relative l2 norm
            l2 = np.linalg.norm((all_xy_u_up[:,2:4] - all_xy_u_up[:,4:6]), axis=0) / np.linalg.norm((all_xy_u_up[:,2:4]), axis=0)
            linf = np.max(np.abs(all_xy_u_up[:,2:4] - all_xy_u_up[:,4:6]), axis=0)
        comm.Bcast(l2,root=0)
        comm.Bcast(linf,root=0)
        return l2, linf


# In[7]:


torch.set_default_dtype(torch.float64)
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")
dtype = torch.float64
pi = torch.tensor(np.pi, device = device)
print(device)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dims = [4,4]
if size != dims[0]*dims[1]:
    raise ValueError("This setting has wrong Cartesian topology.")
cart_comm = comm.Create_cart(dims, periods=[False, False])

# Name File Sample Method
trials = 5
outer_iter = 2000
epochs = 100

L = 4
k = pi * 2**L
theta = pi/3
full_domain = np.array([[0., 0.],
                       [1., 1.]])

mesh = 128
full_dom_dis = [mesh, mesh]
# for each subdomain
n_dom = mesh**2//size
n_bound = int( mesh// ((dims[0]+dims[1])/2) )
n_data  = 0 

test_dis = [256//dims[0], 256//dims[1]]

n_h_layers = 3
n_neurons  = 20
n_output   = 2 # complex values

para_adapt = ParaAdapt(zeta=0.99, omega=0.999, eta=torch.tensor([[1.],[0.01]]).to(device), epsilon=1e-16, epoch_min=50)
methodname = f'helm_complex_l{L}_xy{dims[0]}{dims[1]}_nn{n_h_layers}_{n_neurons}_dom_dis{mesh}'


# In[7]:


l2_norms = []
for trial in range(1, trials+1):
    print("*"*20 + f' run({trial}) '+"*"*20)
    domain, dom_dis  = rank_domain(full_domain, full_dom_dis, rank, size, dims)
    x_dm,y_dm, x_bc,y_bc, x_int,y_int = sampling(domain, dom_dis, dom_dis, dom_dis, rank, size, dims)
    printing_points(x_dm,y_dm, x_bc,y_bc, x_int,y_int, trial, rank)
     
    model      = modelling(domain, n_h_layers, n_neurons, n_output)
    
    q, Lambda, Mu, Mu_max, Bar_v     = paraing()
    optim_change, optimizer, optim_q = optimizers()

    u_adj, dudn_adj, dudt_adj = intering(x_int)
    
    outer_s, mu_s, lambda_s, constr_s, object_s, q_s, loss_s, l2_s, linf_s = voidlist()
    
    # Training loop
    start_time = time.perf_counter()
    for count in range(1, outer_iter+1):
        print("*"*20 + f' outer iteration ({count}) '+"*"*20)
        previous_loss = torch.tensor(torch.inf).to(device)
        for epoch in range(1, epochs + 1):
            def _closure():
                model.eval()
                avg_pde_loss = physics_loss(model,x_dm,y_dm, k)
                avg_bc_loss  = boundary_loss(model,x_bc,y_bc, k, theta)
                avg_int_loss = interface_loss(q, model, x_int,y_int, u_adj, dudn_adj, dudt_adj)
                
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
                print('rank = %d, epoch = %d, loss = %.3e, objective = %.3e, constr_loss = %.3e, %.3e'%(rank, epoch, loss, objective, constr[0], constr[1]))

            with torch.no_grad():
                Bar_v        = para_adapt.zeta*Bar_v + (1-para_adapt.zeta)*constr.pow(2)
                if loss >= para_adapt.omega * previous_loss or epoch == epochs-1:
                    Lambda += Mu * constr
                    Mu_max = torch.max(para_adapt.eta / (torch.sqrt(Bar_v)+para_adapt.epsilon), Mu_max)
                    Mu = torch.min( torch.max( constr/(torch.sqrt(Bar_v)+para_adapt.epsilon), torch.tensor(1.) )*Mu, Mu_max)
                if epoch >= para_adapt.epoch_min:
                    break
            previous_loss = loss.item()
        collect_metrics(count, Mu, Lambda, constr, objective, loss, q,
                            outer_s, mu_s, lambda_s, constr_s, object_s, loss_s, q_s)

        u_adj, dudn_adj, dudt_adj = exchange_interface(model, x_int,y_int, cart_comm, rank, device)
        
        l2, linf = evaluate_write(model, domain, test_dis, methodname, trial, write=False)
        l2_s.append(l2)
        linf_s.append(linf)

        if count%10 ==0:
            l2, linf = evaluate_write(model, domain, test_dis, methodname, trial, write=True)
            if rank == 0:
                print('count = %d, l2 norms (Re, Im) = (%.3e, %.3e)'%(count, l2[0], l2[1]))
            printing(trial, outer_s, mu_s, lambda_s, constr_s, object_s, loss_s, q_s, l2_s, linf_s, rank)
            torch.save(model.state_dict(), f"{methodname}_{trial}_{rank}.pt")
            #contour_prediction(test_dis, trial, dims, size, methodname)
            #plot_update(trial, dims, size, methodname)

    stop_time = time.perf_counter()
    print(f'Rank: {rank}, Elapsed time: {stop_time - start_time:.2f} s')

    # Evaluate model
    l2, linf = evaluate_write(model, domain, test_dis, methodname, trial, write=True)
    l2_norms.append(l2[0])


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


# In[9]:


### post_processing ###
if rank == 0:
    contour_prediction(test_dis, trial2[1], dims, size, methodname)
    plot_update(trial2[1], dims, size, methodname)


# In[ ]:




