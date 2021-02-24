# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:54:25 2021

@author: Ahmed Emam
"""

import sys
import numpy as np
import scipy.io
from pyDOE import lhs
from torch import Tensor, ones, stack, load
from torch.autograd import grad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.append('../')  # PINNFramework etc.
print(sys.path)
import PINNFramework as pf
#'______________________________________________'
#data
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

import pandas as pd
vptxt =np.loadtxt(r'D:\thesis\data_hendrick\2D_gound_models\vp.txt')
x = np.arange(0,10,0.0125)
y = np.arange(0,10,0.0125)
xx, yy = np.meshgrid(x, y, sparse=True)
distance = np.sqrt(yy**2 + xx**2)
time = np.divide(distance,vptxt)#add the velocity file instead of vp txt

data= pd.DataFrame()
u=pd.DataFrame()
t=pd.DataFrame()
data['x']= x
data['y']= y
u= distance

#_______________________________________________________________________________
class BoundaryConditionDataset(Dataset):

    def __init__(self, nb, lb, ub):
        """
        Constructor of the initial condition dataset

        Args:
          n0 (int)
        """
        super(type(self)).__init__()
        time = np.divide(distance,vptxt)
        t = time.flatten()[:, None]
        idx_t = np.random.choice(t.shape[0], 50, replace=False)
        tb = t[idx_t, :]
        self.x_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
        self.x_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
        self.y_lb=  np.concatenate((0*  tb + lb[0],tb), 1)
        self.y_ub=  np.concatenate((0*  tb + ub[0],tb), 1)
        

    def __getitem__(self, idx):
        """
        Returns data for initial state
        """
        ## we didn't activate the idx argument for get item
        return Tensor(self.x_lb).float(), Tensor(self.x_ub).float(),Tensor(self.y_lb).float(),Tensor(self.y_ub).float()

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1


class InitialConditionDataset(Dataset):

    def __init__(self, n0):
        """
        Constructor of the boundary condition dataset

        Args:
          n0 (int)
        """
        super(type(self)).__init__()
        #we need to create the data dateframe first
        
        x = data['x'][:, None]
        y = data['y'][:,None]
        time = np.divide(distance,vptxt)
        t=time.flatten()[:, None]
        u = distance
        idx_x = np.random.choice(x.shape[0], n0, replace=False)
        self.x = x[idx_x, :]
        self.y = y[idx_x, :]
        self.u = u[idx_x, 0:1]
        self.t = np.zeros(self.x.shape)

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1

    def __getitem__(self, idx):
        x = np.concatenate([self.x, self.t], axis=1)
        y = np.concatenate([self.y, self.t], axis=1)
        return Tensor(x).float(), Tensor(y).float()


class PDEDataset(Dataset):
    def __init__(self, nf, lb, ub):
        self.xf = lb + (ub - lb) * lhs(2, nf)
        self.yf = lb + (ub - lb) * lhs(2, nf)

    def __getitem__(self, idx):
        """
        Returns data for initial state
        """
        return Tensor(self.xf).float(), Tensor(self.yf).float()

    def __len__(self):
        """
        There exists no batch processing. So the size is 1
        """
        return 1
if __name__ == "__main__":
    # Domain bounds
    lb = np.array([0.0, 0.0])
    ub = np.array([800.0,0.0])
    # initial condition
    ic_dataset = InitialConditionDataset(n0=50)
    initial_condition = pf.InitialCondition(ic_dataset)
    # boundary conditions
    bc_dataset = BoundaryConditionDataset(nb=50, lb=lb, ub=ub)
    periodic_bc_u = pf.PeriodicBC(bc_dataset, 0, "u periodic boundary condition")
    periodic_bc_u_x = pf.PeriodicBC(bc_dataset, 0, "u_x periodic boundary condition", 1, 0)
    periodic_bc_u_y = pf.PeriodicBC(bc_dataset, 1, "u_y periodic boundary condition", 1, 0)
    # PDE
    pde_dataset = PDEDataset(20000, lb, ub)
    
    def wave2d(x, y, u):
        pred = u
        u = pred[:, 0]
       
        print("x:", x.shape)
        print("u:", v.shape)
        grads = ones(u.shape, device=pred.device) # move to the same device as prediction
        grad_u_x = grad(u, x, create_graph=True, grad_outputs=grads)[0]
        grad_u_y = grad(u, y, create_graph=True, grad_outputs=grads)[0]
        grad_u_t = grad(u, t, create_graph=True, grad_outputs=grads)[0]
       

        # calculate first order derivatives
        u_x = grad_u_x[:, 0]
        u_y = grad_u_y[:,0]
        u_t = grad_u_t[:, 0]
         # calculate second order derivatives
        grad_u_xx=  grad(u_x, x, create_graph=True, grad_outputs=grads)[0]
        grad_u_yy=  grad(u_y, y, create_graph=True, grad_outputs=grads)[0]
        grad_u_tt=  grad(u_t, t, create_graph=True, grad_outputs=grads)[0]
        f =  u_tt-(1*(u_xx+u_yy))
        return f
    pde_loss = pf.PDELoss(pde_dataset, wave2d)
    model = pf.models.MLP(input_size=3, output_size=1, hidden_size=100, num_hidden=4, lb=lb, ub=ub)
    pinn = pf.PINN(model, 2, 2, pde_loss, initial_condition, [periodic_bc_u,periodic_bc_u_x,periodic_bc_u_y], use_gpu= False)
    pinn.fit(50000, 'Adam', 1e-3)
        