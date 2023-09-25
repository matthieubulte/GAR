import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize

def m(x, theta, mu):
    return theta * x + (1-theta)*mu

def sim(N, theta, sig, mu):
    x = np.zeros(N) + mu
    for i in range(1,N):
        x[i] = (1 + sig*np.random.randn()) * m(x[i-1], theta, mu)
    return x

M = 250 # number of replicates
B = 500 # number of bootstrap replicates

phis = [0, 0.05, 0.1, 0.2, 0.5, 1]
Ts = [10, 20, 40, 80, 160, 320, 640]


df = pd.DataFrame(columns=['T', 'phi', 'replicate_id', 'err_mu_hat', 'phi_hat', 'Dt', 'boot_mean_Dt', 'boot_sig_Dt'])

mean = 1.5
noise_var = 0.2

def phi_hat(x, mu_hat, tol=None):
    T = x.shape[0]
    tol = tol or 1.0 / np.sqrt(T)
    def L(phi): return np.array([ (x[j+1] - m(x[j], phi, mu_hat))**2 for j in range(T-1) ]).mean()
    return minimize(L, np.random.rand(), method='Nelder-Mead', bounds=[(0,1)], options=dict(xatol=tol))['x'][0]

def bootstrap_mu_sig(x, B):
    bootstrap = np.zeros(B)
    for b in range(B):
        permed = np.random.permutation(x)
        bootstrap[b] = np.mean(np.power(np.diff(permed), 2))
    
    sig_hat = np.sqrt(bootstrap.var())
    mu_hat = bootstrap.mean()
    
    return mu_hat, sig_hat

run_id = np.random.randint(100000)

for T in Ts:
    for phi in phis:
        print(f"Running T={T} phi={phi}")
        for replicate_id in tqdm(range(M)):
            x = sim(T, phi, noise_var, mean)
            mu_hat = x.mean()
            phi_hat_ = phi_hat(x, mu_hat)
            Dt = np.mean(np.power(np.diff(x), 2))
            boot_mean_Dt, boot_sig_Dt = bootstrap_mu_sig(x, B)
            df.loc[len(df)] = [T, phi, replicate_id, (mean - mu_hat)**2, phi_hat_, Dt, boot_mean_Dt, boot_sig_Dt]

    df.to_csv(f'./results_R_boot_Dt/{run_id}.csv')