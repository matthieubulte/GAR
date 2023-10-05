from load_config import load_config
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import pyfrechet.metric_spaces.wasserstein_1d as W1d
import pandas as pd
from pyfrechet.metric_spaces import MetricData

# todo: change the order of the arguments
def geodesic(x, theta, mu):
    return theta * x + (1-theta)*mu

def bootstrap(x, stat, B, boot_fraction):
    bootstrap = np.zeros(B)
    s = int(np.floor(x.shape[0]) * boot_fraction)
    for b in range(B):
        permed = np.random.permutation(x)[:s, :]
        bootstrap[b] = stat(permed)
    return bootstrap


def iter_wasserstein(T, phi, num_boot, boot_fraction):
    W = W1d.Wasserstein1D()

    def sim(T, phi, mu):
        x = np.zeros((T, mu.shape[0])) + mu
        for i in range(1,T):
            z = geodesic(x[i-1], phi, mu)
            k = (1 - 2*np.random.binomial(1,0.5)) * np.random.random_integers(1, 4)
            x[i, :] = z - np.sin(np.pi*k*z)/np.pi/abs(k)
        return x
    
    def phi_hat(x, tol=None):
        mu_hat = MetricData(W, x).frechet_mean()
        T = x.shape[0]
        tol = tol or 1.0 / np.sqrt(T)
        def L(phi): return np.array([ W._d(x[j+1,:], geodesic(x[j,:], phi, mu_hat))**2 for j in range(T-1) ]).mean()
        return minimize(L, np.random.rand(), method='Nelder-Mead', bounds=[(0,1)], options=dict(xatol=tol))['x'][0]

    def Dt(x):
        return np.array([ W._d(x[j,:], x[j+1,:])**2 for j in range(x.shape[0]-1) ]).mean()

    STD_NORMAL_Q = stats.norm.ppf(W1d.Wasserstein1D.GRID)
    STD_NORMAL_Q[0] = 2*STD_NORMAL_Q[1] - STD_NORMAL_Q[2] # lexp to avoid infs
    STD_NORMAL_Q[-1] = 2*STD_NORMAL_Q[-2] - STD_NORMAL_Q[-3] # lexp to avoid infs
    mean = STD_NORMAL_Q

    x = sim(T, phi, mean)
    mu_hat = MetricData(W, x).frechet_mean()
    
    booted_phi = bootstrap(x, phi_hat, num_boot, boot_fraction)
    booted_dt = bootstrap(x, Dt, num_boot, boot_fraction)

    return W._d(mean, mu_hat)**2, phi_hat(x), Dt(x), booted_phi, booted_dt

def iter_r(T, phi, num_boot, boot_fraction):
    def sim(T, phi, mu):
        sig = 0.2
        x = np.zeros((T,1)) + mu
        for i in range(1,T):
            x[i] = (1 + sig*np.random.randn()) * geodesic(x[i-1], phi, mu)
        return x
    
    def phi_hat(x):
        T = x.shape[0]
        tol = 1.0 / np.sqrt(T)
        mu_hat = x.mean()
        def L(phi): return np.array([ (x[j+1] - geodesic(x[j], phi, mu_hat))**2 for j in range(T-1) ]).mean()
        return minimize(L, np.random.rand(), method='Nelder-Mead', bounds=[(0,1)], options=dict(xatol=tol))['x'][0]
    
    def Dt(x):
        return np.mean(np.power(np.diff(x.flatten()), 2))

    mean = 1.5
    x = sim(T, phi, mean)
    mu_hat = x.mean()

    booted_phi = bootstrap(x, phi_hat, num_boot, boot_fraction)
    booted_dt = bootstrap(x, Dt, num_boot, boot_fraction)

    return (mean - mu_hat)**2, phi_hat(x), Dt(x), booted_phi, booted_dt


print('loading config')
out_name, sim_setting = load_config()
print(sim_setting)

T = int(sim_setting["sample_size"])
phi = sim_setting["phi"]
setup = sim_setting["sim_setup"]
boot_fraction = sim_setting["boot_fraction"]
num_boot = int(sim_setting["num_boot"])

print('running sim')
out_fn = iter_r if setup == 'r' else iter_wasserstein

print('saving result')

err, phi_hat, Dt, booted_phi, booted_dt = out_fn(T, phi, num_boot, boot_fraction)

alphas = np.linspace(0, 1, 50)
Dt_q = np.quantile(booted_dt, alphas)
phi_hat_q = np.quantile(booted_phi, alphas)

sim_setting['result'] = dict(err=err, phi_hat=phi_hat, Dt=Dt, quantiles_Dt=Dt_q, quantiles_phi_hat=phi_hat_q)

sim_setting.to_pickle(out_name)