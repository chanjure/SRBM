#-*-coding: utf-8 -*-

# ----------------------------- #
# Probability density functions #
# ----------------------------- #

"""
	Probability density functions module
	~~~~~~~~~
"""

import numpy as np

def _positive_sigmoid(x):
	"""_positive_sigmoid

	Sigmoid with positive input.
	Subprocess of sigmoid(x)

	Parameters
	----------
	x : np.ndarray
			input values

	Returns
	-------
	np.ndarray
			element-wise sigmoid of input x
	"""

	return 1. / (1. + np.exp(-x))

def _negative_sigmoid(x):
	"""_negative_sigmoid

	Sigmoid with negative input.
	1/(1+e^(-x)) = e^(x)/(1 + e^(x)).
	Subprocess of sigmoid(x)

	Parameters
	----------
	x : np.ndarray
			input values

	Returns
	-------
	np.ndarray
			element-wise sigmoid of input x
	"""

	exp = np.exp(x)
	return exp / (exp + 1.)

def Sigmoid(x):
	"""sigmoid

	Stable sigmoid

	Parameters
	----------
	x : np.ndarray
			input values

	Returns
	-------
	np.ndarray
			element-wise sigmoid of input x
	"""
	
	assert type(x) == np.ndarray, "Err (pdf.sigmoid) : input x must be np.ndarray"

	positive = x >= 0
	negative = ~positive

	result = np.empty_like(x, dtype=np.float64)
	result[positive] = _positive_sigmoid(x[positive])
	result[negative] = _negative_sigmoid(x[negative])

	return result

def Gaussian(x, mean, sig):
  """normal

  Normal distribution centered at x and width sig.

  Parameters
  ----------
  x : np.ndarray
    actual sampled state
  mean : np.ndarray
    mean of Gaussian pdfs of each nodes
  sig : float
    Global std

  Returns
  -------
  np.ndarray
    element-wise Gaussian probability
  """

  return np.exp(-0.5 *(x - mean)*(x - mean)/(sig*sig))/(sig*np.sqrt(2.*np.pi))
