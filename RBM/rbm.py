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

from . import pdf
from . import error

class RBM():
  """Binomial Restricted Boltzmann Machine.

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
  pv, ph = None, None
  w, a, b = None, None, None
  history = {}

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
		"""
    self.n_v = n_v
    self.n_h = n_h
    self._initialize_weights(n_v, n_h)

    self.history['loss'] = []

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

  def fit(self, x, lr, epoch):
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

    N = 1.*x.shape[0]

    for e in range(epoch):

      # Expectation value from the data
      self.ph = pdf.sigmoid(self.b + x@self.w)
      self.h = self._sample(self.ph)

      v_data = np.mean(x, axis=0)
      h_data = np.mean(self.ph, axis=0)
      vh_data = (x.T @ self.ph) / N

      # Expectation value from the Model
      self.pv, self.v, self.ph, self.h = self._CDk(1,x)

      v_model = np.mean(self.pv, axis=0)
      h_model = np.mean(self.ph, axis=0)
      vh_model = (self.pv.T @ self.ph) / N

      # Update parameters
      self.w += lr * (vh_data - vh_model)
      self.a += lr * (v_data - v_model)
      self.b += lr * (h_data - h_model)

      # Print the progress
      loss = self._error(x, self.v, "mse")
      print(e, loss)
      self.history['loss'].append(loss)

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
    pv_n[0] = pdf.sigmoid(self.a + self.h @ self.w.T)
    v_n = np.zeros((k,) + x.shape)
    v_n[0] = self._sample(pv_n[0])

    ph_n = np.zeros((k,) + self.ph.shape)
    ph_n[0] = pdf.sigmoid(self.b + v_n[0] @ self.w)
    h_n = np.zeros((k,) + self.h.shape)
    h_n[0] = self._sample(ph_n[0])

    # k Gibbs sampling
    for i in range(1,k):
      pv_n[i] = pdf.sigmoid(self.a + h_n[i-1] @ self.w.T)
      v_n[i] = self._sample(pv_n[i])

      ph_n[i] = pdf.sigmoid(self.b + v_n[i] @ self.w)
      h_n[i] = self._sample(ph_n[i])

    # Expectation value from k Gibbs samples
    pv_k = np.mean(pv_n, axis=0)
    v_k = np.mean(v_n, axis=0)
    ph_k = np.mean(ph_n, axis=0)
    h_k = np.mean(h_n, axis=0)

    return pv_k, v_k, ph_k, h_k

  def _sample(self, x):
    """Sample

    Sampling methods

    Note
    ----
    We can add some other sampling methods.
		
    Parameters
    ----------
    x : np.ndarray
      The probability density to sample from.
		"""
    # return np.where(np.random.binomial(1,x),1,-1) # {1,-1}
    return np.random.binomial(1,x)  # {1,0}

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
    ph = pdf.sigmoid(self.b + x @ self.w)
    h = self._sample(ph)

    pv = pdf.sigmoid(self.a + h @ self.w.T)
    v = self._sample(pv)

    return pv, ph

