import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 

import numpy as np
import matplotlib.pyplot as plt
from pyfrechet.metric_spaces import MetricData, RiemannianManifold, CorrFrobenius, MetricSpace
from geomstats.geometry.hypersphere import Hypersphere
import pandas as pd
from pyfrechet.metric_spaces.correlation.nearcorr import nearcorr
from tqdm import tqdm
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy import stats
from pyfrechet.metric_spaces import *
import pyfrechet.metric_spaces.wasserstein_1d as W1d


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

M = 250 # number of replicates
B = 500 # number of bootstrap replicates

phis = np.linspace(0, 1, 6)
Ts = [1200]

df = pd.DataFrame(columns=['T', 'phi', 'replicate_id', 'err_mu_hat', 'phi_hat', 'Dt', 'boot_mean_Dt', 'boot_sig_Dt'])

STD_NORMAL_Q = stats.norm.ppf(W1d.Wasserstein1D.GRID)
STD_NORMAL_Q[0] = 2*STD_NORMAL_Q[1] - STD_NORMAL_Q[2] # lexp to avoid infs
STD_NORMAL_Q[-1] = 2*STD_NORMAL_Q[-2] - STD_NORMAL_Q[-3] # lexp to avoid infs
mean = STD_NORMAL_Q

def phi_hat(x, mu_hat):
    T = x.shape[0]
    W = W1d.Wasserstein1D()

    grid = np.linspace(0, 1, 40)

    def calc(phi): return np.array([ W._d(x[j+1,:], gamma(x[j,:], phi, mu_hat))**2 for j in range(T-1) ]).mean()
    errs = np.array([ calc(phi) for phi in grid ])
    
    return grid[np.argmin(errs)]

def bootstrap_mu_sig(x, B):
    bootstrap = np.zeros(B)
    for b in range(B):
        permed = np.random.permutation(np.arange(x.shape[0]))
        bootstrap[b] = np.array([ W._d(x[permed[j],:], x[permed[j+1],:])**2 for j in range(T-1) ]).mean()
    
    sig_hat = np.sqrt(bootstrap.var())
    mu_hat = bootstrap.mean()
    
    return mu_hat, sig_hat


run_id = np.random.randint(100000)

for T in Ts:
    for phi in phis:
        print(f"Running T={T} phi={phi}")
        for replicate_id in tqdm(range(M)):
            x = sim(T, phi, mean)

            W = W1d.Wasserstein1D()
            mu_hat = MetricData(W, x).frechet_mean()

            phi_hat_ = phi_hat(x, mu_hat)

            Dt = np.array([ W._d(x[j,:], x[j+1,:])**2 for j in range(T-1) ]).mean()
            
            boot_mean_Dt, boot_sig_Dt = bootstrap_mu_sig(x, B)

            df.loc[len(df)] = [T, phi, replicate_id, W._d(mean, mu_hat)**2, phi_hat_, Dt, boot_mean_Dt, boot_sig_Dt]

            df.to_csv(f'./{run_id}_results_wasserstein_full.csv')