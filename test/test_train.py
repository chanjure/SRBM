import numpy as np
import pytest

import torch
import os, sys
sys.path.append('../')
import SRBM

def S(field, m):
    N = len(field)
    s = 0.
    for i in range(N):
        s += -0.5*field[i]*(field[(i+1)%N] + field[(i-1)%N] - (2.+ m[i]**2)*field[i])

    return s/N

def S_fast(field,m=2.):
    N = field.shape[0]
    s = m**2 * field**2
    s += 2.*field**2
    s -= field*torch.roll(field,-1)
    s -= field*torch.roll(field,1)
    return torch.sum(0.5*s)/N

# Least square fit
def loss(K, K_phi, Zk, b, r):
    return np.sum((K - Zk*K_phi - b - r)**2)

def dL_dZ(K, K_phi, Zk, b, r):
    return -np.sum(2.*(K - Zk*K_phi - b - r)*K_phi)

def dL_db(K, K_phi, Zk, b, r):
    return -np.sum(2.*(K - Zk*K_phi - b - r))

def dL_dr(K, K_phi, Zk, b, r):
    return -2.*(K - Zk*K_phi - b - r)

def get_scale(model, verbose=True):
    Kin = (-model.sig**2 * (model.w.t() @ model.w)).data.numpy()
    Mss = np.diag((model.m**2).data.numpy())
    K = Kin + Mss
    mu_ = model.m.data.numpy()
    Zk_gd = mu_.min()**2/(2.**2 + 4)

    np.random.seed(1234)
    r_seed = np.random.normal(0.,.01,size=K.shape) # ~ sigma^2
    r_gd = r_seed.T @ r_seed
    b_gd = 0.
    lr = 1e-5

    for i in range(10000):
        l = loss(K, K_phi, Zk_gd, b_gd, r_gd)
    #     print(l)
        dZ = lr * dL_dZ(K, K_phi, Zk_gd, b_gd, r_gd)
        db = lr * dL_db(K, K_phi, Zk_gd, b_gd, r_gd)
        dr = lr * dL_dr(K, K_phi, Zk_gd, b_gd, r_gd)

        Zk_gd -= dZ
        b_gd -= db
        r_gd -= dr
        if i%1000 == 0 and verbose:
            print(i, l, Zk_gd, b_gd)
            
    return Zk_gd, b_gd, r_gd, K

N = 10

# True matrix
m = 2
K_phi = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if i==j:
            K_phi[i][j] = 2 + m**2
        elif (i % N == (j+1) %N) or (i % N == (j-1) %N):
            K_phi[i][j] = -1

@pytest.mark.parametrize("m_scheme", [
    (0),('local'),('global')],
    ids=["0", "local", "global"])
def test_scale(m_scheme):
    init_cond = {'m':3., 'sig':1., 'm_scheme':m_scheme}

    rbm = SRBM.RBM.SRBM(n_v = N, n_h = N, k=10, init_cond=init_cond)

    lr = 1e-1
    epochs = 3000
    batch_size = 64

    init_field = torch.ones((batch_size, N), dtype=torch.double)

    K_phi_tc = torch.DoubleTensor(K_phi)

    history = rbm.unsup_fit(K_phi_tc, S_fast,\
            epochs=epochs, lr=lr, beta=0.5, l2=1e-1, batch_size=batch_size,\
            verbose=True, lr_decay=0.99)

    Zk, b, r, K = get_scale(rbm)

    assert Zk == pytest.approx(1., abs=0.4)

def test_saveload():
    init_cond = {'m':3., 'sig':1., 'm_scheme':'global'}

    rbm = SRBM.RBM.SRBM(n_v = N, n_h = N, k=10, init_cond=init_cond)
    rbm.save('./')
    saved_model = rbm.name
    rbm2 = SRBM.RBM.SRBM(load='./'+saved_model+'.npz')

    assert (rbm2.name == rbm.name) and rbm2.history.keys() == rbm.history.keys()
    assert (rbm.__dict__['_parameters']['w'] == rbm2.__dict__['_parameters']['w']).all().item()
    assert (rbm.__dict__['_parameters']['m'] == rbm2.__dict__['_parameters']['m']).all().item()
    assert (rbm.__dict__['_parameters']['eta'] == rbm2.__dict__['_parameters']['eta']).all().item()
