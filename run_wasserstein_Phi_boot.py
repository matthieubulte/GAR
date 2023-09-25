import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from pyfrechet.metric_spaces import *
import pyfrechet.metric_spaces.wasserstein_1d as W1d
from scipy.optimize import minimize
import sim_config

def gamma(x, theta, mu):
    # the rvs are represented as their quantile functions, so interpolation is euclidean
    return theta * x + (1-theta)*mu

def sim(N, theta, mu):
    x = np.zeros((N, mu.shape[0])) + mu
    for i in range(1,N):
        z = gamma(x[i-1], theta, mu)

        k = (1 - 2*np.random.binomial(1,0.5)) * np.random.random_integers(1, 4)
        x[i, :] = z - np.sin(np.pi*k*z)/np.pi/abs(k)
    return x


df = pd.DataFrame(columns=['T', 'phi', 'replicate_id', 'err_mu_hat', 'phi_hat', 'phi_boot'])

STD_NORMAL_Q = stats.norm.ppf(W1d.Wasserstein1D.GRID)
STD_NORMAL_Q[0] = 2*STD_NORMAL_Q[1] - STD_NORMAL_Q[2] # lexp to avoid infs
STD_NORMAL_Q[-1] = 2*STD_NORMAL_Q[-2] - STD_NORMAL_Q[-3] # lexp to avoid infs
mean = STD_NORMAL_Q

def phi_hat(x, mu_hat, tol=None):
    T = x.shape[0]
    tol = tol or 1.0 / np.sqrt(T)
    def L(phi): return np.array([ W._d(x[j+1,:], gamma(x[j,:], phi, mu_hat))**2 for j in range(T-1) ]).mean()
    return minimize(L, np.random.rand(), method='Nelder-Mead', bounds=[(0,1)], options=dict(xatol=tol))['x'][0]


def bootstrap_phi_hat(x, B):
    W = W1d.Wasserstein1D()
    bootstrap = np.zeros(B)
    s = int(np.floor(x.shape[0]) * sim_config.S)
    for b in range(B):
        permed = np.random.permutation(x)
        mu_hat = MetricData(W, permed)[:s].frechet_mean()
        bootstrap[b] = phi_hat(permed, mu_hat)
    return bootstrap


run_id = np.random.randint(100000)


for T in sim_config.Ts:
    for phi in sim_config.phis:
        print(f"Running T={T} phi={phi}")
        for replicate_id in tqdm(range(sim_config.M)):
            x = sim(T, phi, mean)
            W = W1d.Wasserstein1D()
            mu_hat = MetricData(W, x).frechet_mean()
            phi_hat_ = phi_hat(x, mu_hat)
            boot = bootstrap_phi_hat(x, sim_config.B)
            df.loc[len(df)] = [T, phi, replicate_id, W._d(mean, mu_hat)**2, phi_hat_, [boot]]

        df.to_csv(f'./results_wasserstein_boot_phi/{run_id}.csv')