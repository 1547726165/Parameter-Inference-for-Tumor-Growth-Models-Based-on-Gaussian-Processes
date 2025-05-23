from numpy.core.numeric import Inf
import torch
import time
import numpy as np
import scipy.stats
from pyDOE import *
from scipy.optimize import minimize
import GP_processing
from scripts import HMC
from scipy.optimize import dual_annealing
from scipy.optimize import basinhopping
from scipy.optimize import rosen, shgo
from scipy.optimize import rosen, differential_evolution
torch.set_default_dtype(torch.double)

class MAGI_PDE_Infer(object):
    def __init__(self, True_Model, KL=False):
        self.PDE_Model = True_Model
        self.KL= KL
        self.pde_operator=True_Model.pde_operator
        self.aIdx = True_Model.aIdx
        self.noisy_known=True_Model.noisy_known
        self.x_I=True_Model.x_I
        self.n_I, self.d = self.x_I.size()
        self.n_bound = True_Model.n_bound
        self.y_obs = True_Model.y_obs
        self.n_obs, self.p = self.y_obs.size()
        self.x_obs = True_Model.x_obs
        self.y_bound=True_Model.y_bound
        self.x_bound=True_Model.x_bound
        self.y_all=True_Model.y_all
        self.x_all=True_Model.x_all
        self.GP_Components=True_Model.GP_Components
        self.GP_Models=True_Model.GP_Models
        #self.GP_PDE_Components=True_Model.GP_PDE_Components

    def map(self, nEpoch = 2500):
        u_KL, u, Lu_GP, Lu_GP_KL, GP_Trans_u_Lu = self._Pre_Process()
        time0 = time.time()
        # optimize the initial theta
        para_theta = self.PDE_Model.para_theta
        d_theta = para_theta.shape[0]
        self.d_theta = d_theta

        if self.PDE_Model.source_term == 0 or  self.PDE_Model.source_term==1:
            u_current = u
            current_opt = np.Inf
            current_theta=para_theta
            self.u_KL_initial = u_KL
            x = torch.randn(2+u_KL.shape[0])
            u_KL_censored = x[2:2+u_KL.shape[0]]

        if self.PDE_Model.source_term == 0 or  self.PDE_Model.source_term==1: 
            u_lr= 5e-1
            para_theta = torch.rand(d_theta).double()
            if self.cheat ==1 or self.cheat ==2: para_theta = self.PDE_Model.theta_true.clone()
        u_KL=u_KL.requires_grad_()
        para_theta=para_theta.requires_grad_()
        p = self.y_obs.shape[1]
        lognoisescale = torch.zeros(p)
        for i in range(p):
            lognoisescale[i]=torch.log(self.GP_Components[i]['noisescale'].double())
        if self.noisy_known is False  :
            lognoisescale=lognoisescale.requires_grad_()
            self.optimizer_u_theta = torch.optim.LBFGS([u_KL,para_theta,lognoisescale], lr = u_lr)
        else :
            self.optimizer_u_theta = torch.optim.LBFGS([u_KL,para_theta], lr = u_lr)
            lognoisescale_opt=lognoisescale
        #self.optimizer_u_theta = torch.optim.SGD([u,para_theta], lr = u_lr, momentum=0.8)
        #pointwise_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_u_theta, step_size=500, gamma=0.95)
        pointwise_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_u_theta, lr_lambda = lambda epoch: 1/((epoch+1)**0.5), last_epoch=-1)
        print('start optimiza theta and u:')

        U_KL_trace = ()
        def closure():
            self.optimizer_u_theta.zero_grad()
            loss = self.Minus_Log_Posterior(u_KL, para_theta, lognoisescale / 2)
            loss.backward()
            return loss
        for epoch in range(nEpoch):
            self.optimizer_u_theta.zero_grad()
            loss_u_theta = self.Minus_Log_Posterior(u_KL, para_theta, lognoisescale / 2)
            if epoch==0:
                loss_u_theta_opt=loss_u_theta.clone().detach()
                u_KL_opt=u_KL.clone().detach()
                theta_opt=para_theta.clone().detach()
                if self.noisy_known is False : lognoisescale_opt=lognoisescale.clone().detach()
            else:
                #if para_theta[0]<0: para_theta[0] = torch.abs(para_theta[0])
                if loss_u_theta<loss_u_theta_opt:
                    loss_u_theta_opt=loss_u_theta.clone().detach()
                    u_KL_opt=u_KL.clone().detach()
                    theta_opt=para_theta.clone().detach()
                    if self.noisy_known is False : lognoisescale_opt=lognoisescale.clone().detach()
            loss_u_theta.backward()
            self.optimizer_u_theta.step(closure)
            pointwise_lr_scheduler.step()
            if (np.isnan(self.Minus_Log_Posterior(u_KL, para_theta, lognoisescale / 2).detach().numpy())):
                u_KL = u_KL_opt
                para_theta = theta_opt
                if self.noisy_known is False : lognoisescale=lognoisescale_opt
            if (epoch+1) % 500 == 0 :
                #print(para_theta)
                print(epoch+1, '/', nEpoch, 'current opt: theta:', theta_opt.numpy(),'error/out_scale', torch.exp(lognoisescale_opt).clone().detach().numpy()/self.GP_Components[0]['outputscale'])
                #print('current state: theta:', para_theta.clone().detach().numpy(),'error/out_scale', torch.exp(lognoisescale).clone().detach().numpy()/self.GP_Components[0]['outputscale'])
                print('gradient', torch.mean(torch.abs(u_KL.grad.squeeze())).numpy(), para_theta.grad.numpy())
                #'loss:', loss_u_theta.clone().detach().numpy(),
            U_KL_trace = U_KL_trace + (u_KL_opt,)
        print(loss_u_theta)
        u_KL.requires_grad_(False)
        para_theta.requires_grad_(False)
        lognoisescale.requires_grad_(False)

        p = self.y_obs.shape[1]
        sigma_e_sq_MAP = torch.zeros(p)
        for i in range(p):
            lognoisescale[i]=torch.log(self.GP_Components[i]['noisescale'].double())
            self.GP_Components[0]['noisescale'] = torch.max (torch.exp(lognoisescale_opt[i]), 1e-6 * self.GP_Components[i]['outputscale'])
            sigma_e_sq_MAP[i] = self.GP_Components[i]['noisescale']
        u_opt = torch.empty(self.n_I,self.p).double()
        
        for i in range(self.p):
            u_opt[:,i] = GP_Trans_u_Lu[i]['u_mean_I'] + GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL_opt[:,i]
            #u_opt[:,i] = self.GP_Components[i]['mean'] + GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL_opt[:,i]
        theta_err=(torch.mean((theta_opt-self.PDE_Model.theta_true).square())).sqrt().clone()
        theta_err_relative=(torch.mean(((theta_opt-self.PDE_Model.theta_true)/self.PDE_Model.theta_true).square())).sqrt().clone()
        print('Estimated parameter:', (theta_opt.clone().detach()).numpy(), 'True parameter:',self.PDE_Model.theta_true.numpy(), 'Error of theta:', theta_err,'relative err',theta_err_relative)

        time1 = time.time() - time0
        map_est={
            'u_KL_trace':U_KL_trace,
            'theta_err': theta_err, 
            'theta_err_relative':theta_err_relative,
            'sigma_e_sq_MAP': sigma_e_sq_MAP, 
            'theta_MAP' : theta_opt.clone().detach(), 
            'u_MAP' : u_opt, 
            'u_KL_MAP': u_KL_opt,
            'time': time1
        }
        return (map_est)
    
    def Loss_Theta_Marginal(self, theta, u_current, Lu_GP_KL, GP_Trans_u_Lu, Lu_GP, pde_parameter=4):

        if self.pde_operator==0:
            Lu_PDE = self.PDE_Model.Source(self.x_I, u_current, theta)
            Lu_Error = Lu_PDE - Lu_GP
            lkh =  torch.mean(torch.square(Lu_Error[:,1]))
            theta_loss = lkh.detach()
            return(theta_loss.numpy()) 

    def Minus_Log_Posterior(self, u_KL, para_theta, logsigma = None):
        if self.pde_operator == 0 :
            u = torch.empty(self.n_I,self.p).double()
            Lu_GP= torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
                Lu_GP[:,i] = self.GP_Trans_u_Lu[i]['Lu_mean_I']  + self.GP_Trans_u_Lu[i]['u_to_Lu_GP'] @ (u[:,i%2]-self.GP_Components[i%2]['mean'])
            Lu_PDE = self.PDE_Model.Source(self.x_I, u, para_theta)
            lkh = torch.zeros((1, 3))
            #print(torch.mean((u[:,1]-self.V_test).square()))
            lkh[0,0] = -0.5 * torch.sum(torch.square(u_KL))# - (para_theta[0]<0) * 1e6 * torch.exp(- 100* para_theta[0])
            outputscale = self.GP_Components[0]['outputscale']    
            noisescale = self.noisy_known* (self.GP_Components[0]['noisescale'].clone()) + (1-self.noisy_known) * torch.max(torch.exp(2 * logsigma[0]), 1e-6 * outputscale)
            lkh[0,1] = -0.5 / noisescale * torch.sum ( torch.square(u[self.aIdx[0:self.n_obs-self.n_bound],0]-self.y_obs[0:self.n_obs-self.n_bound,0])) - 0.5 * (self.n_obs-self.n_bound) * torch.log(noisescale) -0.5 / (1e-6*self.GP_Components[0]['outputscale']) * torch.sum ( torch.square(u[self.aIdx[self.n_obs-self.n_bound:self.n_obs],0]-self.y_obs[self.n_obs-self.n_bound:self.n_obs,0]))
            Lu_Error = Lu_PDE - Lu_GP
            lkh[0,2] =  -0.5 * torch.cat((Lu_Error[:,0],Lu_Error[:,1])) @ self.LKL_U_inv @ torch.cat((Lu_Error[:,0],Lu_Error[:,1])).T /self.GP_Components[0]['outputscale']
            lkh[:,1] = 2 * lkh[:,1] * self.n_I/self.n_obs
        return (-torch.sum(lkh))
    
    def Sample_Using_HMC(self, n_epoch = 5000, lsteps=100, epsilon=1e-5, n_samples=20000, Map_Estimation = None, Normal_Approxi_Only = False):
        if Map_Estimation == None : Map_Estimation=self.map(nEpoch = n_epoch)
        self.Map_Estimation = Map_Estimation
        log_sigma=torch.log(Map_Estimation['sigma_e_sq_MAP']).double() / 2
        u_KL=Map_Estimation['u_KL_MAP']
        theta=Map_Estimation['theta_MAP']
        self.sampler = HMC.Posterior_Density_Inference(self.Minus_Log_Posterior, u_KL, theta, log_sigma, u_KL.shape, theta.shape, log_sigma.shape, noisy_known=self.noisy_known, lsteps=lsteps, epsilon=epsilon, n_samples=n_samples)
        self.Normal_Approxi_Only = Normal_Approxi_Only
        self.Posterior_PDE_NA = self.sampler.Normal_Approximation()

        if Normal_Approxi_Only is True:
            theta_CI_NA=self.Posterior_Summary()
            print('Confidence Interval:',theta_CI_NA)
            return (self.Posterior_PDE_NA, self.Map_Estimation)
        self.HMC_sample = self.sampler.Sampling()
        theta_CI, theta_CI_NA=self.Posterior_Summary(draw_posterior_sample = True)
        print('Confidence Interval:',theta_CI)
        return (self.HMC_sample, self.Posterior_PDE_NA, self.Map_Estimation)
    
    def Posterior_Summary(self, alpha =0.05, draw_posterior_sample = False):
        Var = torch.diag(self.Posterior_PDE_NA['variance'])
        Mean = self.Posterior_PDE_NA['mean']
        mean_theta = Mean[-(self.d_theta+1):-1]
        var_theta = Var[-(self.d_theta+1):-1]
        theta_CI_NA = torch.zeros(2, mean_theta.shape[0])
        theta_CI_NA[0,:] = mean_theta - scipy.stats.norm.cdf(alpha)*torch.sqrt(var_theta)
        theta_CI_NA[1,:] = mean_theta + scipy.stats.norm.cdf(alpha)*torch.sqrt(var_theta)
        if self.Normal_Approxi_Only is True :
            return (theta_CI_NA)
        chain=self.HMC_sample['samples']
        sampler=self.sampler
        x_pred=self.PDE_Model.x_pred
        GP_Trans_u_Lu = self.GP_Trans_u_Lu
        N_chain = chain.shape[0]
        Err_u_chain = torch.zeros(N_chain)
        u_pred = torch.zeros((N_chain, self.n_I,self.p))
        theta_post = torch.zeros((N_chain, sampler.theta_shape[0]))
        sigmasq_post = torch.zeros((N_chain, 1, self.p))
        for s in range (N_chain):
            sample = torch.tensor(chain[s,:])
            u_KL_current, theta_current, logsigma_current = sampler.devectorize(sample, sampler.u_KL_shape, sampler.theta_shape, sampler.sigma_shape)
            u_KL_current = u_KL_current.clone()
            if draw_posterior_sample is True:
                u_current = torch.empty(self.n_I,self.p).double()
                for i in range(self.p):
                    u_current[:,i] = GP_Trans_u_Lu[i]['u_mean_I'] + GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL_current[:,i]
                #Err_u_chain[s], u_pred_temp = self.Predict_err(u_current)[0:2]
                u_pred[s] = u_current.clone()
            theta_post[s,:] = theta_current.clone()
            sigmasq_post[s,:] = torch.exp(2 * logsigma_current).clone()
        self.posterior_summary = {
            'u_pred': u_pred,
            'u_pred_true': self.PDE_Model.y_pred,
            'theta_posterior':theta_post,
            'sigmasq_posterior':sigmasq_post,
            'u_err_posterior':Err_u_chain
        }
        burn_in = int(N_chain*0.1)
        theta_post = theta_post[burn_in:N_chain,:]
        theta_CI = self.Credible_Interval(theta_post, alpha = 0.05)
        return (theta_CI, theta_CI_NA)

    def Credible_Interval(self, samples, alpha = 0.05):
        N = samples.shape[0]
        d = samples.shape[1]
        L = int(N*alpha/2)
        U = N - int(N*alpha/2)
        CI = torch.zeros(2, d)
        CI[0,:]=samples.sort(0).values[L,:]
        CI[1,:]=samples.sort(0).values[U,:]
        return (CI)


    def _Pre_Process(self):
        # obtain features from GP_Components
        
        if self.pde_operator == 0 : 
            nu=self.PDE_Model.nu
            y_obs_all = self.y_obs#torch.cat((self.y_obs,self.y_bound),0)
            n_obs_all = y_obs_all.shape[0]
            x_obs_all = self.x_obs#torch.cat((self.x_obs,self.x_bound),0)

            GP_model = self.GP_Models[0]
            nugget_gp = torch.cat((GP_model.noisescale/GP_model.outputscale *torch.ones(self.n_obs-self.n_bound),1e-6 * torch.ones(self.n_bound))) 
            kernel = GP_model.kernel
            C = kernel.K(x_obs_all) + torch.diag(nugget_gp)
            U = GP_model.mean + kernel.K(self.x_I,x_obs_all) @ torch.linalg.inv(C) @ (y_obs_all[:,0]-GP_model.mean)
            C = kernel.K(self.x_I) + 1e-6 * torch.eye(self.n_I)
            Lap_U = kernel.LK(self.x_I)[0] @ torch.linalg.inv(C) @ (U-GP_model.mean)
            U_t = kernel.LK(self.x_I)[1] @ torch.linalg.inv(C) @ (U-GP_model.mean)



            U = U.reshape(-1,1)
            Lap_U = Lap_U.reshape(-1,1)
            U_t = U_t.reshape(-1,1)

            GP_model1=GP_processing.GP_modeling(self.PDE_Model, noisy = False, nu=nu, noisy_known=True)
            GP_model1.Train_GP(self.PDE_Model,self.x_I, Lap_U, noisy = False)
            self.GP_Components.append({
                #'aIdx':aIdx, # non-missing data index
                'mean':GP_model1.mean,
                'kernel':GP_model1.kernel,
                'outputscale':GP_model1.outputscale,
                'noisescale':GP_model1.noisescale
            })
            self.GP_Models.append(GP_model1)

            U = torch.cat((U,Lap_U,U_t),1)
            self.p = 2
            GP_Trans_u_Lu = []
            u_KL = torch.empty(self.n_I, self.p).double()
            Lu_GP = torch.empty(self.n_I, self.p).double()
            Lu_GP_KL = torch.empty(self.n_I, self.p).double()
            M_KL_all = torch.zeros(self.p)
            M_KL_Lu_all = 0
            for i in range(self.p):
                base_u = i % 2
                base_ope = int(i/2)
                kernel = self.GP_Components[i]['kernel']
                outputscale = self.GP_Components[i]['outputscale']
                # Compute GP prior covariance matrix
                K_II = kernel.K(self.x_I, self.x_I) + 1e-6 * torch.eye(self.n_I)
                u_mean_I = self.GP_Components[i]['mean'] 
                # dimension reduction via KL expansion
                P, V, Q = torch.svd(K_II)
                if self.KL is False :
                    M_u_KL = self.n_I
                else:
                    #M=sum(V >2 *1e-6)
                    M_u_KL = sum(np.cumsum(V) < (1-1e-6)* torch.sum(V))
                    print('number of KL basis:', M_u_KL)
                Trans_u_to_u_KL = torch.diag(torch.pow(V, -1/2)) @ P.T
                Trans_u_KL_to_u = P @ torch.diag(torch.pow(V , 1/2))
                Trans_u_to_u_KL = Trans_u_to_u_KL[0:M_u_KL,:] / np.sqrt(outputscale)#u_to_u_KL
                Trans_u_KL_to_u = Trans_u_KL_to_u[:,0:M_u_KL] * np.sqrt(outputscale)#u_KL_to_u
                if i ==0 : Trans_u_KL_to_u_0 = Trans_u_KL_to_u
                '''
                L_chol = torch.linalg.cholesky(K_II)
                Trans_u_to_u_KL = torch.linalg.inv(L_chol)
                Trans_u_KL_to_u = L_chol
                '''
                K_inv = torch.linalg.inv(K_II)
                self.GP_Models[i].K_inv = K_inv
                # obtain initial values and PDE information
                u_KL[0:M_u_KL,i] = Trans_u_to_u_KL @ (U[:,i] - u_mean_I)
                Lu_mean_I = 0
                LK_II = self.GP_Components[base_u]['kernel'].LK(self.x_I)[base_ope]
                Trans_u_to_Lu = LK_II @ self.GP_Models[base_u].K_inv
                Trans_u_KL_to_Lu = Trans_u_to_Lu @ Trans_u_KL_to_u_0
                Lu_GP[:,i] = Lu_mean_I + Trans_u_to_Lu @ (U[:,base_u] - self.GP_Components[base_u]['mean']) 
                GP_Trans_u_Lu.append({
                    'u_mean_I' : u_mean_I,
                    'Lu_mean_I' : Lu_mean_I,
                    'u_KL_to_u' : Trans_u_KL_to_u ,
                    'u_to_u_KL' : Trans_u_to_u_KL , 
                    'u_KL_to_Lu_GP' : Trans_u_KL_to_Lu, 
                    'u_to_Lu_GP' : Trans_u_to_Lu, 
                    })
                M_KL_all[i] = M_u_KL
            #u_KL = u_KL[0:M_KL_all,:].clone().detach()
            self.U_test = U[:,0].clone()

            self.GP_Trans_u_Lu = GP_Trans_u_Lu
            kernel1 = self.GP_Components[0]['kernel']
            LKL_all = kernel1.LKL(self.x_I)[4]
            LK_all = kernel1.LK(self.x_I)[2]
            KL_all = kernel1.KL(self.x_I)[2]
            self.LKL_U = LKL_all - LK_all @ self.GP_Models[0].K_inv @ KL_all + 1e-6 * torch.eye(2*self.n_I)
            self.LKL_U_inv = torch.linalg.inv(self.LKL_U)
            self.LKL_U_margin = kernel1.LKL(self.x_I)[1] - kernel1.LK(self.x_I)[1] @ self.GP_Models[0].K_inv @ kernel1.KL(self.x_I)[1] + 1e-6 * torch.eye(self.n_I)
            self.LKL_U_inv_margin = torch.linalg.inv(self.LKL_U_margin)
        
            return (u_KL, U, Lu_GP, Lu_GP_KL, GP_Trans_u_Lu)

        
    def Loss_for_Censored_Component(self, x):
        u_KL = self.u_KL_initial
        x = torch.tensor(x)
        d_theta = self.PDE_Model.para_theta.shape[0]
        theta = x[0:d_theta]
        u_KL_censored = x[d_theta:d_theta + u_KL.shape[0]]

        if self.pde_operator == 0 :
            u_KL[:,1] = u_KL_censored
            u = torch.empty(self.n_I,self.p).double()
            Lu_GP= torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
                Lu_GP[:,i] = self.GP_Trans_u_Lu[i]['Lu_mean_I']  + self.GP_Trans_u_Lu[i]['u_to_Lu_GP'] @ (u[:,i%2]-self.GP_Components[i%2]['mean'])
            Lu_PDE = self.PDE_Model.Source(self.x_I, u, theta)
            lkh = torch.zeros((1, 3))
            lkh[0,0] = -0.5 * torch.sum(torch.square(u_KL)) 
            Lu_Error = Lu_PDE - Lu_GP
            lkh[0,2] =  -0.5 * torch.cat((Lu_Error[:,0],Lu_Error[:,1])) @ self.LKL_U_inv @ torch.cat((Lu_Error[:,0],Lu_Error[:,1])).T /self.GP_Components[0]['outputscale']
            #lkh[0,2] =  lkh[0,2] -0.5 * torch.cat((Lu_Error[:,1],Lu_Error[:,3])) @ self.LKL_V_inv @ torch.cat((Lu_Error[:,1],Lu_Error[:,3])).T /self.GP_Components[1]['outputscale']
            #print(torch.mean(torch.square(self.V_test-u[:,1])))
        return (torch.sum(lkh))