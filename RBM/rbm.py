#-*-coding: utf-8 -*-

# ------------------------------ #
# Scalar field Boltzmann Machine #
# ------------------------------ #

"""
	Scalar field Boltzmann Machine core module
	~~~~~~~~~
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.set_default_tensor_type(torch.DoubleTensor) 
torch.set_default_dtype(torch.float64)
# Float in python is equal to double in C
# https://docs.python.org/3/library/stdtypes.html


class SRBM(nn.Module):
  """Scalar field Restricted Boltzmann Machine.

  Parameters
  ----------
  """
  name = ''
  history = {}

  def __init__(self, n_v=None, n_h=None, k=None, \
			fixed=None, init_cond=None, \
			name='SRBM', load=False):
    super(SRBM, self).__init__()

    # Use GPU if available
    self.device = torch.device("cuda:%d"%(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")

    self.history['loss'] = []
    self.history['w'] = []
    self.history['m'] = []
    self.history['eta'] = []
    self.history['dw'] = []
    self.history['dm'] = []
    self.history['S'] = []
    self.history['S_std'] = []

    # Get timetag for the model
    if load:
      print("Loading model from "+load)
      mz = np.load(load, allow_pickle=True)
      
      self.name=mz['name'].item()

      self.n_v = mz['n_v'].item()
      self.n_h = mz['n_h'].item()
      self.k = mz['k'].item()

      self.w = nn.Parameter(torch.DoubleTensor(mz['w']))
      self.m = nn.Parameter(torch.DoubleTensor(mz['m']))
      self.eta = nn.Parameter(torch.DoubleTensor(mz['eta']))
      self.sig = mz['sig'].item()
      
      self.m_scheme = mz['m_scheme'].item()
      self.history = mz['history'].item()

    else:
      time_tag = time.strftime("%y%m%d_%H%M%S", time.gmtime())
      self.name += name + '-' + time_tag
      print("Initializing model "+self.name)
    
      self.n_v = n_v
      self.n_h = n_h
      self.k = k

      # Default init scheme
      try:
        w_sig = init_cond['w_sig']
      except:
        w_sig = 0.1

      try:
        w = init_cond['w']
        self.w = nn.Parameter(w)
      except:
        self.w = nn.Parameter(torch.randn(n_h,n_v)*w_sig)
		
      try:
        mu = init_cond['m']
      except:
        mu = 1.

      try:
        self.m_scheme = init_cond['m_scheme']
      except:
        self.m_scheme = 0
  
      self.m = nn.Parameter(mu*torch.ones(n_v))

      # self.eta = torch.randn(n_h)
      self.eta = nn.Parameter(torch.zeros(n_h))

      if init_cond != None:
        self.sig = init_cond['sig']
      else:
        self.sig = 1.

  def sample_from_p(self, p, std):
    return torch.normal(p, std)

  def v_to_h(self, v):
    p_h = F.linear(self.sig**2 * v, self.w, self.eta)
    sample_h = self.sample_from_p(p_h, self.sig)
    return p_h, sample_h

  def h_to_v(self, h):
    p_v = F.linear(h, self.w.t())/self.m.pow(2)
    sample_v = self.sample_from_p(p_v, 1./self.m)
    return p_v, sample_v

  def forward(self, v, k = None):
    if k == None:
      k = self.k

    p_h_, h_ = self.v_to_h(v)

    for _ in range(k):
      p_v_, v_ = self.h_to_v(h_)
      p_h_, h_ = self.v_to_h(v_)

    return p_v_, v_, p_h_, h_, v

  def get_grad(self, v):
    N = v.shape[0]
    dw = (self.sig**2 * F.linear(F.linear(v, self.w).t(), v.t()))/N \
        + torch.einsum('ij,k->ikj', v, self.eta).mean(0)
		
    if self.m_scheme == 'local':
      dm = (-v.pow(2)*self.m).mean(0)
    elif self.m_scheme == 'global':
      dm = -self.m*(v.pow(2).sum(1)).mean(0)
    else:
      dm = torch.zeros(self.n_v).to(self.device)

    #deta = F.linear(v, self.w).mean(0)
    deta = torch.zeros(self.n_h).to(self.device)

    return dw, deta, dm

  def free_energy(self, v):
    phi_w = F.linear(v, self.w)

    mass_term = -0.5*(v.pow(2)*self.m.pow(2)).sum(1)
    kin_term = 0.5*self.sig**2 * phi_w.pow(2).sum(1)
    bias_term = F.linear(self.eta, phi_w)
    return (mass_term + kin_term + bias_term).mean()/self.n_v
  
  def unsup_fit(self, K_true, S, epochs, lr, batch_size=64, verbose=True, lr_decay=0):
    data = torch.ones((batch_size, self.n_v)).to(self.device)
    for epoch in range(epochs):
      p_v, data, _, _, v = self.forward(data)

      #loss = self.free_energy(data) + S(data, 2.).mean()
      #loss = -S(data, 2.).mean()
      # rev KL = -S_rbm + S_true - constant
      #        = free_energy + phi K_true phi

      with torch.no_grad():
        data = data.detach()
        S_density = 0.5* torch.trace(data @ K_true @ data.t())/batch_size/self.n_v
        loss = self.free_energy(data) + S_density
      
        if epoch == 0:
          K_inv = torch.linalg.inv(K_true)
        
        dw_d = self.w @ K_inv
        deta_d = torch.zeros(self.n_h).to(self.device)

        if self.m_scheme == 'local':
          dm_d = -F.linear(self.m, K_inv)
        elif self.m_scheme == 'global':
          dm_d = -self.m*torch.trace(K_inv)
        else:
          dm_d = torch.zeros(self.n_v).to(self.device)
        
        dw_m, deta_m, dm_m = self.get_grad(data)

        dw = dw_d - dw_m
        deta = deta_d - deta_m
        dm = dm_d - dm_m

        self.w += lr*dw
        self.eta += lr*deta
        self.m += lr*dm
    
        self.loss = torch.mean(loss.detach()).cpu().numpy()
        self.dw = dw
        self.dm = dm
        self.outstr = "epoch :%d "%(epoch)
        if epoch % batch_size == 0:
          if verbose:
            ver_ = True
          else:
            ver_ = False

          if lr_decay:
            lr *= lr_decay
          
          self.outstr += 'lr: %.5f '%(lr)  
        
        self.history['S'].append(S_density.detach().cpu().numpy())
        self._historian(ver_)
        ver_ = False

    return self.history

  def fit(self, train_dl, epochs, lr, verbose=True, lr_decay=0):
    for epoch in range(epochs):
      loss_ = torch.empty((len(train_dl),train_dl.batch_size))
      for i, data in enumerate(train_dl):
        data = Variable(data[0].view(-1,self.n_v)).to(self.device, non_blocking=True)

        p_v, v_, _, _, v = self.forward(data)
        S_density = self.free_energy(v_)
        loss_[i] = self.free_energy(v) - S_density

        with torch.no_grad():
          dw_d, deta_d, dm_d = self.get_grad(v)
          dw_m, deta_m, dm_m = self.get_grad(v_)

          dw = dw_d - dw_m
          deta = deta_d - deta_m
          dm = dm_d - dm_m

          self.w += lr*dw
          self.eta += lr*deta
          self.m += lr*dm
        
      with torch.no_grad():
        if lr_decay:
          lr *= lr_decay
              
        self.loss = np.mean(loss_.detach().cpu().numpy())
        self.dw = dw
        self.dm = dm
        self.outstr = "epoch :%d "%(epoch)
        self.outstr += 'lr: %.5f '%(lr)  
        self.history['S'].append(S_density.detach().cpu().numpy())
        self._historian(verbose)

    return self.history

  def painter(self, save=None):
    """painter

    Draw learning curve, svd of weight, learned pattern.

    Parameters
    ----------
    save : str
      figure directory
    """

    t = len(self.history['loss'])
    fig, ax_l = plt.subplots()
    
    # Loss plot
    ax_l.plot(np.arange(t), self.history['loss'], 'C3.-',\
        label = 'loss %0.3f'%(self.history['loss'][-1]))
    ax_l.set_xlabel('Epochs', fontsize=15)
    ax_l.set_ylabel('Loss (rmse)', fontsize=15) # ('Loss (%s)')%([name of loss type])

    # w svd mode plot
    # Do svd
    print("Doing SVD")
    s_hist = np.zeros((t,self.n_h))
    for i in range(t):
      _, s, _ = np.linalg.svd(self.history['w'][i])
      s_hist[i] = s.copy()
    print("Done")

    ax_a = ax_l.twinx()
    for i in range(self.n_h):
      ax_a.plot(np.arange(t), s_hist.T[i])
    ax_a.set_ylabel(r'$\omega_{\alpha}$', fontsize=15)
    
    ax_l.grid(True)
    ax_l.legend(loc='upper center')
    fig.suptitle(self.name, fontsize=18)

    if save:
      fig.savefig(save+self.name+'.jpg', dpi=125)

    plt.show()
    plt.clf()

  def _historian(self, verbose=True):
    self.history['loss'].append(self.loss)
    self.history['w'].append(self.w.detach().cpu().numpy().copy())
    self.history['m'].append(self.m.detach().cpu().numpy().copy())
    self.history['eta'].append(self.eta.detach().cpu().numpy().copy())
    self.history['dw'].append(self.dw.detach().cpu().numpy().copy())
    self.history['dm'].append(self.dm.detach().cpu().numpy().copy())
    self.outstr += 'loss : %.5f'%(self.loss)

    if verbose:
      print(self.outstr)

  def save(self, fpath):
    np.savez(fpath+'/'+self.name+'.npz', name=self.name, n_v=self.n_v, n_h=self.n_h, k=self.k,\
            w=self.w.detach().numpy(), m=self.m.detach().numpy(), eta=self.eta.detach().numpy(),\
            sig=self.sig, m_scheme=self.m_scheme, history=self.history, allow_pickle=True)
