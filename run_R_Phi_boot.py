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
S = 0.7 # size of bootstrap sample

# phis = np.linspace(0, 1, 6)
phis = [0, 0.05, 0.1, 0.2, 0.5, 1]
# Ts = [50, 100, 250, 500, 1000, 1200]
Ts = [10, 20, 40, 80, 160, 320, 640]


df = pd.DataFrame(columns=['T', 'phi', 'replicate_id', 'err_mu_hat', 'phi_hat', 'phi_boot'])

mean = 1.5
noise_var = 0.2

def phi_hat(x, mu_hat, tol=None):
    T = x.shape[0]
    tol = tol or 1.0 / np.sqrt(T)
    def L(phi): return np.array([ (x[j+1] - m(x[j], phi, mu_hat))**2 for j in range(T-1) ]).mean()
    return minimize(L, np.random.rand(), method='Nelder-Mead', bounds=[(0,1)], options=dict(xatol=tol))['x'][0]

def bootstrap_phi_hat(x, B):
    bootstrap = np.zeros(B)
    s = int(np.floor(x.shape[0]) * S)
    for b in range(B):
        permed = np.random.permutation(x)[:s]
        mu_hat = permed.mean()
        bootstrap[b] = phi_hat(permed, mu_hat)
    return bootstrap

run_id = np.random.randint(100000)

for T in Ts:
    for phi in phis:
        print(f"Running T={T} phi={phi}")
        for replicate_id in tqdm(range(M)):
            x = sim(T, phi, noise_var, mean)
            mu_hat = x.mean()
            phi_hat_ = phi_hat(x, mu_hat)
            boot = bootstrap_phi_hat(x, B)
            df.loc[len(df)] = [T, phi, replicate_id, (mean - mu_hat)**2, phi_hat_, [boot]]

        df.to_csv(f'./results_R_boot_phi/{run_id}.csv')