#-*-coding: utf-8 -*-

# ---------------------------- #
# Restricted Boltzmann Machine #
# ---------------------------- #

"""
	Restricted Boltzmann Machine core module
	~~~~~~~~~
"""

import numpy as np
import matplotlib.pyplot as plt
import abc

from . import pdf
from . import error

class RBM(metaclass=abc.ABCMeta):
  """Restricted Boltzmann Machine.

  Core module of rbm package.

  Attributes
  ----------
  v : np.ndarray
    Visible states
  h : np.ndarray
    Hidden states
  pv : np.ndarray
    Probability of visible state P(v|h)
  ph : np.ndarray
    Probability of hidden state P(h|v)
  sig_v : float
    Standard deviation of visible state
  sig_h : float
    Standard deviation of hidden state
  w : np.ndarray
    Interaction matrix between visible state and hidden state
  a : np.ndarray
    External field for visible state
  b : np.ndarray
    External field for hidden state
  history : dict
    dictionary contaning training history.
    ex : loss, likelyhood, w^2, d(w)

  """

  v, h = None, None
  n_v, n_h = None, None
  pv, ph = None, None
  sig_v, sig_h = None, None
  w, a, b = None, None, None
  loss = 0.
  history = {}

  def __setattr__(self, var, val):
    if hasattr(self, var):
      super().__setattr__(var, val)
    else:
      raise AttributeError("Err (setattr) : Setting new attribute is forbiden.")

  def __init__(self, n_v, n_h):
    """Initializing

    Initializes attributes

    Note
    ----
		
    Parameters
    ----------
    n_v : int
      Number of visible states.
    n_h : int
      Number of hidden states.
    mode : string
      Two letter mode of RBM.
      B : Bernoulli, G : Gaussian
    args : dict
      Additional arguments.
		"""
  
    self.n_v = n_v
    self.n_h = n_h
    self._initialize_weights(n_v, n_h)

    self.v = np.zeros(n_v)
    self.pv = np.zeros(n_v)

    self.h = np.zeros(n_h)
    self.ph = np.zeros(n_h)

  def _initialize_weights(self, n_v, n_h):
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
		"""

    self.w = np.random.normal(scale=0.01, size=(n_v,n_h))
    self.a = np.zeros(n_v)
    self.b = -4.0 * np.ones(n_h)

  def fit(self, x, lr, epoch, CDk=1):
    """fit

    Fit the model paramters

    Note
    ----
    In the future, more features may be added

    Parameters
    ----------
    x : np.ndarray
      Training data with the shape of (number of data, dim1, dim2, ... )
    lr : float
      Learning rate
    epoch : int
      Number of epochs
    """

    error.typeErr(self.fit.__name__, x, np.ndarray)

    for e in range(epoch):

      # Expectation value from the data
      self.h, self.ph = self._sample(self.b + x@self.w)
      v_data, h_data, vh_data = self._getGrad(x)
      #print('h_data',h_data)

      # Expectation value from the Model
      self.pv, self.v, self.ph, self.h = self._CDk(CDk,x)
      v_model, h_model, vh_model = self._getGrad()
      #print('h_model',h_model)

      # Update parameters
      self.w += lr * (vh_data - vh_model)
      self.a += lr * (v_data - v_model)
      self.b += lr * (h_data - h_model)
      #print('w',self.w)

      # Print progress and generate history
      self.loss = self._error(x, self.v, "mse")
      print(e, self.loss)
      self._historian()

  @abc.abstractmethod
  def _historian(self):
    """_historian

    Generate history

    """
    pass

  @abc.abstractmethod
  def _getGrad(self, x=None):
    """_getGrad

    Get positive and negative gradient

    Parameters
    ----------
    x : np.ndarray
      Input data for positive gradient
    """
    pass

  def _CDk(self, k, x):
    """Constrative Divergence k

    Constrative Divergence using k steps of Gibbs sampling

    Note
    ----
		
    Parameters
    ----------
    k : int
      Number of Gibbs sampling.
    x : np.ndarray
      Initial visible data.
		"""
    pv_n = np.zeros((k,) + x.shape) # (Gibbs step, N data, n_v, 1)
    v_n = np.zeros((k,) + x.shape)
    v_n[0], pv_n[0] = self._sample(self.a + self.h @ self.w.T)

    ph_n = np.zeros((k,) + self.h.shape)
    h_n = np.zeros((k,) + self.h.shape)
    h_n[0], ph_n[0] = self._sample(self.b + v_n[0] @ self.w)

    # k Gibbs sampling
    for i in range(1,k):
      v_n[i], pv_n[i] = self._sample(self.a + h_n[i-1] @ self.w.T)
      h_n[i], ph_n[i] = self._sample(self.b + v_n[i] @ self.w)

    # Expectation value from k Gibbs samples
    pv_k = np.mean(pv_n, axis=0)
    v_k = np.mean(v_n, axis=0)

    ph_k = np.mean(ph_n, axis=0)
    h_k = np.mean(h_n, axis=0)

    if k > 1:
      self.sig_v = np.std(v_k, axis=0)
      self.sig_h = np.std(h_k, axis=0)

    return pv_k, v_k, ph_k, h_k

  @abc.abstractmethod
  def _sample(self, x):
    """Sample

    Sampling methods

    Note
    ----
    We can add some other sampling methods.
		
    Parameters
    ----------
    sid : string
      pdf to sample from
    x : np.ndarray
      The probability density to sample from.
      In case of Gaussian it is a center of the distribution.
    sig : float
      In case of Gaussian, standard deviation.
		"""
    pass

  def _error(self, x, v, e_ftn):
    """Error

    Error functions

    Note
    ----
    We can add some other error functions.
		
    Parameters
    ----------
    x : np.ndarray
      Resampled pattern.
    v : np.ndarray
      True pattern.
    e_ftn : string
      error function type
		"""
    
    if e_ftn == 'mse':
      return np.mean((x - v)**2)

    else:
      raise ValueError("Err (_error) : No matching error function type.")

  @abc.abstractmethod
  def reconstruct(self, x):
    """Reconstruct

    Reconstruction from the model.

    Note
    ----
    Use probability density in output because we don't want sampling noise in reconstruction.
		
    Parameters
    ----------
    x : np.ndarray
      Input pattern.
		"""
    pass

class BBRBM(RBM):
  """Binomial Restricted Boltzmann Machine.
  """

  def __init__(self, n_v, n_h):
    super().__init__(n_v, n_h)  

  def _initialize_weights(self, n_v, n_h):
    super()._initialize_weights(n_v, n_h)

    self.history['loss'] = [] # MSE Loss 
    self.history['w'] = [] # weight matrix

  def _historian(self):
    self.history['loss'].append(self.loss)
    self.history['w'].append(self.w.copy())

  def _getGrad(self, x = None):

    N = 1.*self.v.shape[0]

    if type(x) == np.ndarray:
      v = np.mean(x, axis=0)
      h = np.mean(self.ph, axis=0)
      vh = (x.T @ self.ph) / N
    elif type(x) == type(None):
      v = np.mean(self.pv, axis=0)
      h = np.mean(self.ph, axis=0)
      vh = (self.pv.T @ self.ph) / N
    else:
      raise ValueError("Err (_getGrad) : Check input data x.")

    return v, h, vh


  def _sample(self, x):
    # Bernoulli (or Binomial)
    # return np.where(np.random.binomial(1,x),1,-1) # {1,-1}
    
    proba = pdf.Sigmoid(x)
    state = np.random.binomial(1,proba)
    
    return state, proba

  def reconstruct(self, x):

    h, ph = self._sample(self.b + x @ self.w)
    v, pv = self._sample(self.a + h @ self.w.T)

    return pv, ph

class GGRBM(RBM):
  """Gaussian Restricted Boltzmann Machine.
  """
  
  def __init__(self, n_v, n_h, sig_v = 0.001, sig_h = 0.001):
    super().__init__(n_v, n_h)
    self.sig_v = sig_v
    self.sig_h = sig_h
    
    self.history['loss'] = []
    self.history['w'] = []
    self.history['sig_v'] = []
    self.history['sig_h'] = []

  def _initialize_weights(self, n_v, n_h):
    super()._initialize_weights(n_v, n_h)
  
  def _historian(self):
    self.history['loss'].append(self.loss)
    self.history['w'].append(self.w.copy())
    self.history['sig_v'].append(self.sig_v)
    self.history['sig_h'].append(self.sig_h)
  
  def _getGrad(self, x = None):

    N = 1.*self.v.shape[0]

    # Data expectation value
    if type(x) == np.ndarray:
      v = np.mean(x, axis=0)
      h = np.mean(self.h, axis=0)
      vh = (x.T @ self.h) / N
    # Model expectation value
    elif type(x) == type(None):
      v = np.mean(self.v, axis=0)
      h = np.mean(self.h, axis=0)
      vh = (self.v.T @ self.h) / N
    else:
      raise ValueError("Err (_getGrad) : Check input data x.")

    return v, h, vh

  def _sample(self, x):

    if x.shape[1] == self.n_v:
      sig = self.sig_v
    elif x.shape[1] == self.n_h:
      sig = self.sig_h
    else:
      raise ValueError(r"Err (_sample) : $\sigma$ is not properly set.")

    mean = sig*sig*x

    state = np.random.normal(loc=mean,scale=sig)
    # One sigma probability - Could be better
    #proba = pdf.Gaussian(state + 0.5*sig,mean,sig)*sig
    proba = np.array([1.])

    # Gaussian
    return state, proba

  def reconstruct(self, x):

    h, ph = self._sample(self.b + x @ self.w)
    v, pv = self._sample(self.a + h @ self.w.T)

    return v, h
