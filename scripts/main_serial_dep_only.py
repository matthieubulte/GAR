from load_config import load_config
from serial_dep_method import prop_test
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import pyfrechet.metric_spaces.wasserstein_1d as W1d
import pandas as pd
from pyfrechet.metric_spaces import MetricData

# todo: change the order of the arguments
def geodesic(x, theta, mu):
    return theta * x + (1-theta)*mu


def iter_wasserstein(T, phi, num_boot, boot_fraction):
    W = W1d.Wasserstein1D()

    def sim(T, phi, mu):
        x = np.zeros((T, mu.shape[0])) + mu
        for i in range(1,T):
            z = geodesic(x[i-1], phi, mu)
            k = (1 - 2*np.random.binomial(1,0.5)) * np.random.random_integers(1, 4)
            x[i, :] = z - np.sin(np.pi*k*z)/np.pi/abs(k)
        return x
    
    STD_NORMAL_Q = stats.norm.ppf(W1d.Wasserstein1D.GRID)
    STD_NORMAL_Q[0] = 2*STD_NORMAL_Q[1] - STD_NORMAL_Q[2] # lexp to avoid infs
    STD_NORMAL_Q[-1] = 2*STD_NORMAL_Q[-2] - STD_NORMAL_Q[-3] # lexp to avoid infs
    mean = STD_NORMAL_Q

    x = sim(T, phi, mean)
    
    D_mat = np.zeros((T,T))
    for i in range(T):
        for j in range(i+1,T):
            D_mat[i,j] = W._d(x[i,:], x[j,:])
            D_mat[j,i] = D_mat[i,j]
    stat_CM, booted_CM, stat_KS, booted_KS = prop_test(D_mat, B=num_boot)

    return stat_CM, stat_KS, booted_CM, booted_KS

def iter_r(T, phi, num_boot, boot_fraction):
    def sim(T, phi, mu):
        sig = 0.2
        x = np.zeros((T,1)) + mu
        for i in range(1,T):
            x[i] = (1 + sig*np.random.randn()) * geodesic(x[i-1], phi, mu)
        return x
    
    mean = 1.5
    x = sim(T, phi, mean)
    D_mat = np.zeros((T,T))
    for i in range(T):
        for j in range(i+1,T):
            D_mat[i,j] = np.abs(x[i,0] - x[j,0])
            D_mat[j,i] = D_mat[i,j]

    stat_CM, booted_CM, stat_KS, booted_KS = prop_test(D_mat, B=num_boot)

    return stat_CM, stat_KS, booted_CM, booted_KS


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

stat_CM, stat_KS, booted_CM, booted_KS = out_fn(T, phi, num_boot, boot_fraction)

alphas = np.linspace(0, 1, 50)
CM_q = np.quantile(booted_CM, alphas)
KS_q = np.quantile(booted_KS, alphas)

sim_setting['result'] = dict(
    CM=stat_CM,
    KS=stat_KS,
    quantiles_CM=CM_q,
    quantiles_KS=KS_q)

sim_setting.to_pickle(out_name)