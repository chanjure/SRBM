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

torch.set_default_tensor_type(torch.DoubleTensor) 
# Float in python is equal to double in C
# https://docs.python.org/3/library/stdtypes.html

class SRBM:
  """Scalar field Restricted Boltzmann Machine.

  Parameters
  ----------
  """
  name = ''
  history = {}

  def __init__(self, n_v, n_h, k, \
			fixed=None, init_cond=None, \
			name='SRBM', load=False):
    """Initializing

    Initializes attributes

    Note
    ----
		
    Parameters
    ----------
    name : str
      Name of the model.
      Or if load=True the path of the saved model.
    n_v : int
      Number of visible states.
    n_h : int
      Number of hidden states.
    sig_h : float
      Width of the auxiliary states.
    load : Bool
      Load the saved model.
		"""

    self.history['loss'] = []
    self.history['w'] = []
    self.history['m'] = []
    self.history['eta'] = []
    self.history['dw'] = []

    # Get timetag for the model
    if load:
      print("Loading model from "+name)
      mz = np.load(name)
      
      self.name=mz.name
      
      self.n_v = mz.n_v
      self.n_h = mz.n_h
      self.k = k

      self.w = mz.w
      self.m = mz.mass
      self.eta = mz.eta
      self.sig = mz.sig
      
      self.history = mz.history

    else:
      time_tag = time.strftime("%y%m%d_%H%M", time.gmtime())
      self.name += name + '-' + time_tag
      print("Initializing model "+self.name)
    
      self.n_v = n_v
      self.n_h = n_h
      self.k = k

      self._initialize_weights(n_v, n_h, init_cond)

  def _initialize_weights(self, n_v, n_h, init_cond=None):
    """Initializing

    Initializes weights

    Note
    ----
    Different schemes may be chosen.

    The following ones are from Hinton (2010).
    
    Parameters
    ----------
    n_v : int
      Number of visible states.
    n_h : int
      Number of hidden states.
    sig : float
      standard derivation of initial weight
    """

    # Default init scheme
    self.w = nn.Parameter(torch.randn(n_h,n_v))
		
    if init_cond != None:
      mu = init_cond['m']
    else:
      mu = 1.
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

    p_h, h = self.v_to_h(v)
    h_ = h

    for _ in range(k):
      p_v_, v_ = self.h_to_v(h_)
      p_h_, h_ = self.v_to_h(v_)

    return p_v_, v_, p_h_, h_, v

  def get_grad(self, v):
    N = v.shape[0]
    dw = (self.sig**2 * F.linear(F.linear(v, self.w).t(), v.t()))/N \
        + torch.einsum('ij,k->ikj', v, self.eta).mean(0)
		
    #dm = (-v.pow(2)*self.mass).mean(0)
    dm = 0.
    #deta = F.linear(v, self.w).mean(0)
    deta = 0.

    return dw, deta, dm

  def free_energy(self, v):
    phi_W = F.linear(v, self.w)

    mass_term = -0.5*(v.pow(2)*self.m.pow(2)).sum(1)
    kin_term = 0.5*self.sig**2 * phi_W.pow(2).sum(1)
    bias_term = F.linear(self.eta, phi_W)
    return (mass_term + kin_term + bias_term).mean()

  def fit(train_dl, epoches, lr):
    for epoch in range(epochs):
      loss_ = []
      for _, data in enumerate(train_dl):
        data = Variable(data[0].view(-1,N))

        p_v, v_, _, _, v = self.forward(data)
        loss = self.free_energy(v) - self.free_energy(v_)
        loss_.append(loss.data)

        with torch.no_grad():
          dw_d, deta_d, dm_d = self.get_grad(v)
          dw_m, deta_m, dm_m = self.get_grad(v_)

          dw = dw_d - dw_m
          deta = deta_d - deta_m
          dm = dm_d - dm_m

          self.w += lr*dw
          self.eta += lr*deta
          self.m += lr*dm
			
      self.loss = np.mean(loss_)
      self.dw = dw.copy()
      self.outstr = "epoch :%d "%(epoch)
      self._historian()

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
    self.history['w'].append(self.w.data.numpy().copy())
    self.history['m'].append(self.m.data.numpy().copy())
    self.history['eta'].append(self.eta.data.numpy().copy())
    self.history['dw'].append(self.dw.data.numpy().copy())
    self.outstr += 'loss : %.5f'%(self.loss)

    if verbose:
      print(self.outstr)
