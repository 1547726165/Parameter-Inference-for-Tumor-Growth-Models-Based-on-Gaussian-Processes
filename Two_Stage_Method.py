from numpy.core.numeric import Inf
import torch
import time
import numpy as np
import scipy.stats
from pyDOE import *
from scipy.optimize import minimize

import GP_processing

from scripts import HMC
torch.set_default_dtype(torch.double)

class Two_Stage_Method(object):
    def __init__(self, True_Model):
        self.PDE_Model = True_Model
        self.pde_operator=True_Model.pde_operator
        self.aIdx = True_Model.aIdx
        self.noisy_known=True_Model.noisy_known
        self.x_I=True_Model.x_I
        self.n_I, self.d = self.x_I.size()
        self.n_bound = True_Model.x_bound.shape[0]
        self.n_all=self.n_I+self.n_bound
        self.y_obs = True_Model.y_obs
        self.n_obs, self.p = self.y_obs.size()
        self.x_obs = True_Model.x_obs
        self.y_bound=True_Model.y_bound
        self.x_bound=True_Model.x_bound
        self.y_all=True_Model.y_all
        self.x_all=True_Model.x_all
        self.PDE_Model = True_Model
        self.GP_Components=True_Model.GP_Components
        self.GP_Models=True_Model.GP_Models
        #self.GP_PDE_Components=True_Model.GP_PDE_Components

    def mean_square_loss(self, theta, u_dirivatives, pde_operator):
        u_dynamic = self.dynamic(theta, u_dirivatives, pde_operator)
        if pde_operator ==0: err = torch.mean(torch.square(u_dirivatives[:,2]-u_dynamic))
        return (err)

    def minimize_msl(self, pde_operator = None, sigma_e = None):
        time0 = time.time()
        if pde_operator is None : pde_operator = self.pde_operator
        u_dirivatives = self.Lu_estimation(self.y_obs, self.x_obs, self.PDE_Model.sigma_e)
        current_opt = np.Inf
        current_theta = self.PDE_Model.para_theta
        d_theta = self.PDE_Model.para_theta.shape[0]
        theta_cand=lhs(d_theta, samples=d_theta*5, criterion='maximin')
        for ini in range(d_theta*5):
            res = minimize(self.mean_square_loss, (torch.tensor(theta_cand[ini,:])), args=(u_dirivatives, pde_operator), method='Nelder-Mead', options={'ftol': 1e-6})
            if res['fun']<current_opt: 
                current_theta=res['x']
                current_opt=res['fun']
        theta_err = (torch.mean((torch.tensor(current_theta)-self.PDE_Model.theta_true).square())).sqrt().clone()
        theta_err_relative=(torch.mean(((torch.tensor(current_theta)-self.PDE_Model.theta_true)/self.PDE_Model.theta_true).square())).sqrt().clone()
        print('Two-Stage method:',current_theta,'error', theta_err)

        time1 = time.time() - time0
        self.Two_Stage_Estimation = {
            'theta_err': theta_err,
            'theta_err_relative':theta_err_relative,
            'theta_ts': torch.tensor(current_theta),
            'u_dirivatives_ts': u_dirivatives,
            'time': time1
        }
        return (self.Two_Stage_Estimation)

    def Lu_estimation(self, y_obs, x_obs, sigma_e):
        if self.pde_operator == 0: 
            nu=self.PDE_Model.nu
            y_obs_all = self.y_obs#torch.cat((self.y_obs,self.y_bound),0)
            n_obs_all = y_obs_all.shape[0]
            x_obs_all = self.x_obs#torch.cat((self.x_obs,self.x_bound),0)
            GP_model = self.GP_Models[0]
            nugget_gp = torch.cat((GP_model.noisescale/GP_model.outputscale*torch.ones(self.n_obs-self.n_bound),1e-6 * torch.ones(self.n_bound))) 
            kernel = GP_model.kernel
            C = kernel.K(x_obs_all) + torch.diag(nugget_gp)
            U_to_U = kernel.K(self.x_I,x_obs_all)
            U = GP_model.mean + U_to_U @ torch.linalg.inv(C) @ (y_obs_all[:,0]-GP_model.mean)
            C = kernel.K(self.x_I) + 1e-6 * torch.eye(self.n_I)
            U_to_Lap_U = kernel.LK(self.x_I)[0]
            Lap_U = U_to_Lap_U @ torch.linalg.inv(C) @ (U-GP_model.mean)
            U_to_U_t = kernel.LK(self.x_I)[1]
            U_t = U_to_U_t @ torch.linalg.inv(C) @ (U-GP_model.mean)
            U = U.reshape(-1,1)
            Lap_U = Lap_U.reshape(-1,1)
            U_t = U_t.reshape(-1,1)
            U = torch.cat((U,Lap_U,U_t),1)
            return (U)

    def dynamic(self, theta, u_dirivatives, pde_operator):
        if pde_operator ==0:
            u_dynamic = self.PDE_Model.Source(self.x_I, u_dirivatives, theta)
            u_dynamic = u_dynamic[:,1]
        return (u_dynamic)



