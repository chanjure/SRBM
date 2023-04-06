# Train Scalar Field RBM.

import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import os, sys
sys.path.append('../')

from SRBM.RBM import rbm

plot_dir = 'images/'
model_dir = 'models/'

# Calculate action
def S(field, m):
    N = len(field)
    s = 0.
    for i in range(N):
        s += -0.5 * field[i] * \
                (field[(i + 1) % N] + field[(i - 1) % N] - (2. + m[i]**2) * field[i])

    return s/N

# Calculate action for torch tensors
def S_fast(field, m):
    N = field.shape[0]
    s = m**2 * field**2
    s += 2.*field**2
    s -= field*torch.roll(field, -1)
    s -= field*torch.roll(field, 1)
    return 0.5*torch.sum(s)/N

def main():
    N = 10
    
    # Calculate true kernel
    m = 2.
    K_phi = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i==j:
                K_phi[i][j] = 2. + m**2
            elif (i % N == (j + 1) % N) or (i % N == (j - 1) % N):
                K_phi[i][j] = -1.

    # One choice of Cholesky solution for m=5.
    W_phi = np.linalg.cholesky(-K_phi + np.diag([5**2]*N))

    # Set random seed for reproducibility.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Train the model.
    # init_cond = {'w':torch.DoubleTensor(W_phi.copy()), 'm':5., 'sig':1., 'm_scheme':'local'}
    init_cond = {'m':3., 'sig':1., 'm_scheme':0}

    # Initialize SRBM
    #rbm = SRBM.RBM.SRBM(n_v=N, n_h=N, k=10, init_cond=init_cond)
    model = rbm.SRBM(n_v=N, n_h=N, k=10, init_cond=init_cond)

    # Training parameters
    lr = 1e-1
    epochs = 3000
    batch_size = 64

    # Initial configuration
    # Cold start
    init_field = torch.ones((batch_size, N), dtype=torch.double)

    # Train the model
    history = model.unsup_fit(K_phi, S_fast, \
            epochs=epochs, lr=lr, batch_size=batch_size, \
            verbose=True, lr_decay=0.99)
    
    # Plot the results
    # Learning curve
    plt.plot(np.arange(len(history['loss'])),history['loss'])
    plt.title('KL')
    plt.xlabel('epoch')
    plt.ylabel(r'$\mathcal{L}$')
    plt.savefig(plot_dir+model.name+'_lc.pdf')
    
    # Gradient
    plt.plot(np.arange(len(history['dw'])), \
            np.mean(np.mean(history['dw'], axis=1), axis=1), \
            label='mean')
    plt.plot(np.arange(len(history['dw'])), \
            np.min(np.min(history['dw'], axis=1),axis=1), \
            label='min')
    plt.plot(np.arange(len(history['dw'])), \
            np.max(np.max(history['dw'], axis=1),axis=1), \
            label='max')
    plt.title('dW')
    plt.xlabel('epoch')
    plt.ylabel(r'$\frac{d L}{dw}$')
    plt.legend()
    plt.savefig(plot_dir+model.name+'_lc.pdf')

    # Reconstructed action distribution
    n_samples = 1000
    recon_field = torch.ones((n_samples, N))
    v_pred, v_, h_pred, _, v = model.forward(recon_field,n_samples)
    n_samples = len(v)

    S_pred = np.zeros(n_samples)
    S_K = np.zeros(n_samples)
    M = m*np.ones(N)

    for i in range(n_samples):
        S_pred[i] = S(v_.data.numpy()[i],M)
        S_K[i] = -model.free_energy(v_[i:i+1]).data.numpy()/N

    plt.hist(S_K, bins=50, density=True, \
            color='C2', label='Model', alpha=0.8)
    plt.hist(S_pred, bins=50, density=True, \
            color='C1', label='Reconstructed', alpha=0.8)    

    plt.legend()
    plt.title('Action histogram')
    plt.savefig(plot_dir+model.name+'_S.pdf')

    # SVD of coupling matrix squared
    s_hist = np.zeros((epochs,N))
    for i in range(epochs):
        _, s_, _ = np.linalg.svd(history['w'][i])
        s_hist[i] = s_

    plt.plot(np.arange(epochs),s_hist**2)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel(r'$w_{\alpha}^2$')
    plt.title(r'$w^2$ evolution')
    plt.savefig(plot_dir+model.name+'_w.pdf')

    # Last few steps
    plt.plot(np.arange(epochs)[-10:],s_hist[-10:]**2, '.-')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel(r'$w_{\alpha}^2$')
    plt.title(r'$w^2$ evolution')
    plt.savefig(plot_dir+model.name+'_w_last.pdf')

    # Kernel SVD values
    s_hist = np.zeros((epochs,N))
    mu2 = np.diag(np.ones(N))

    for i in range(epochs):
        WW_ = history['w'][i].T@history['w'][i]
        K_ = -model.sig**2 * WW_ + np.diag(history['m'][i]**2)
        if i ==0:
            K_i = K_.copy()
        s_ = np.sort(np.linalg.eigvals(K_))
        s_hist[i] = s_

    plt.plot(np.arange(epochs),s_hist)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel(r'$K_{\alpha}$')
    plt.title('K eigenvalue')
    plt.savefig(plot_dir+model.name+'_K.pdf')

    # Last few steps
    plt.plot(np.arange(epochs)[-10:],s_hist[-10:], '.-')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel(r'$K_{\alpha}$')
    plt.title('K eigenvalue')
    plt.savefig(plot_dir+model.name+'_K_last.pdf')

    # Mass parameter of the model
    plt.plot(np.arange(len(history['m'])),history['m'])
    plt.axhline(np.sqrt(m**2 + 2. + 2.), ls='--', color='C3', \
            label='Minimum Cholesky mass limit')
    plt.title('Mass evolution')
    plt.xlabel('epoch')
    plt.ylabel('mass')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir+model.name+'_mass.pdf')

    # Last few steps
    plt.plot(np.arange(len(history['m']))[-10:],history['m'][-10:], '.-')
    plt.title('Mass evolution')
    plt.xlabel('epoch')
    plt.ylabel('mass')
    plt.grid(True)
    plt.savefig(plot_dir+model.name+'_mass_last.pdf')

    # K_rbm off diagonal part
    Kin = (-model.sig**2 * (model.w.t() @ model.w)).data.numpy()
    Mss = np.diag((model.m**2).data.numpy())
    K = Kin + Mss

    K_off = K - np.diag(np.diag(K))
    plt.imshow(K_off, cmap='gray', vmax=K_off.max(), vmin=K_off.min())
    plt.colorbar()
    plt.title('K off diagonal')
    plt.savefig(plot_dir+model.name+'_K_offdiagonal.pdf')

    # Coupling matrix as image
    w_rbm = model.w.data.numpy()
    plt.imshow(w_rbm, cmap='gray')
    plt.colorbar()
    plt.title('RBM coupling matrix')
    plt.savefig(plot_dir+model.name+'_W_img.pdf')

    # Coupling matrix off-diagonal
    w_rbm = model.w.data.numpy()
    plt.imshow(w_rbm - np.diag(np.diag(w_rbm)), cmap='gray')
    plt.colorbar()
    plt.title('W off-diagonal part')
    plt.savefig(plot_dir+model.name+'_W_offdiagonal.pdf')

if __name__ == '__main__':
    main()
