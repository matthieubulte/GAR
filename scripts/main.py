from load_config import load_config
from serial_dep_method import prop_test
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import pyfrechet.metric_spaces.wasserstein_1d as W1d
from pyfrechet.metric_spaces import MetricData
from pyfrechet.metric_spaces import log_cholesky

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


def iter_log_cholesky(T, phi, num_boot, boot_fraction):
    M = log_cholesky.LogCholesky(10)

    def log_chol_to_L(DL):
        n = DL.shape[0]
        d = int((-1 + np.sqrt(1 + 8 * n)) / 2)
        L = np.zeros(shape=(d,d))
        L[np.diag_indices(d)] = np.exp(DL[:d])
        L[np.tril_indices(d, -1)] = DL[d:]
        return L

    def noise(x):
        C_eps = np.zeros_like(x)
        C_eps[:M.dim] = .2 * np.random.normal(size=M.dim)
        C_eps[M.dim:] = .5 * np.random.normal(size=x.shape[0] - M.dim)
        
        L_eps = log_chol_to_L(C_eps)
        x = log_cholesky.log_chol_to_spd(x)
        return log_cholesky.spd_to_log_chol(L_eps.dot(x).dot(L_eps.T))

    def sim(T, phi, mu):
        x = np.zeros((T, mu.shape[0])) + mu
        for i in range(1,T):
            x[i, :] = noise(geodesic(x[i-1], phi, mu))
        return x
    
    def phi_hat(x, tol=None):
        mu_hat = MetricData(M, x).frechet_mean()
        T = x.shape[0]
        tol = tol or min(1.0 / T, 1e-4)
        def L(phi): return np.array([ M._d(x[j+1,:], geodesic(x[j,:], phi, mu_hat))**2 for j in range(T-1) ]).mean()
        return minimize(L, np.random.rand(), method='Nelder-Mead', bounds=[(0,1)], options=dict(xatol=tol))['x'][0]

    def Dt(x):
        return np.array([ M._d(x[j,:], x[j+1,:])**2 for j in range(x.shape[0]-1) ]).mean()

    mean = log_cholesky.spd_to_log_chol(np.eye(M.dim))

    x = sim(T, phi, mean)
    mu_hat = MetricData(M, x).frechet_mean()
    
    booted_phi = bootstrap(x, phi_hat, num_boot, boot_fraction)
    booted_dt = bootstrap(x, Dt, num_boot, boot_fraction)

    D_mat = np.zeros((T,T))
    for i in range(T):
        for j in range(i+1,T):
            D_mat[i,j] = M._d(x[i,:], x[j,:])
            D_mat[j,i] = D_mat[i,j]
    stat_CM, booted_CM, stat_KS, booted_KS = prop_test(D_mat, B=num_boot)

    return M._d(mean, mu_hat)**2, phi_hat(x), Dt(x), stat_CM, stat_KS, booted_phi, booted_dt, booted_CM, booted_KS

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
        tol = tol or min(1.0 / T, 1e-4)
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

    D_mat = np.zeros((T,T))
    for i in range(T):
        for j in range(i+1,T):
            D_mat[i,j] = W._d(x[i,:], x[j,:])
            D_mat[j,i] = D_mat[i,j]
    stat_CM, booted_CM, stat_KS, booted_KS = prop_test(D_mat, B=num_boot)

    return W._d(mean, mu_hat)**2, phi_hat(x), Dt(x), stat_CM, stat_KS, booted_phi, booted_dt, booted_CM, booted_KS

def iter_r(T, phi, num_boot, boot_fraction):
    def sim(T, phi, mu):
        sig = 0.2
        x = np.zeros((T,1)) + mu
        for i in range(1,T):
            x[i] = (1 + sig*np.random.randn()) * geodesic(x[i-1], phi, mu)
        return x
    
    def phi_hat(x, tol=None):
        T = x.shape[0]
        tol = tol or min(1.0 / T, 1e-4)
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

    D_mat = np.zeros((T,T))
    for i in range(T):
        for j in range(i+1,T):
            D_mat[i,j] = np.abs(x[i,0] - x[j,0])
            D_mat[j,i] = D_mat[i,j]

    stat_CM, booted_CM, stat_KS, booted_KS = prop_test(D_mat, B=num_boot)

    return (mean - mu_hat)**2, phi_hat(x), Dt(x), stat_CM, stat_KS, booted_phi, booted_dt, booted_CM, booted_KS


print('loading config')
out_name, sim_setting = load_config()
print(sim_setting)

T = int(sim_setting["sample_size"])
phi = sim_setting["phi"]
setup = sim_setting["sim_setup"]
boot_fraction = sim_setting["boot_fraction"]
num_boot = int(sim_setting["num_boot"])

print('running sim')
if setup == 'r':
    out_fn = iter_r
elif setup == 'wasserstein':
    out_fn = iter_wasserstein
elif setup == 'log_cholesky':
    out_fn = iter_log_cholesky
else:
    raise Exception(f"Unknown setup: {setup}")

print('saving result')

err, phi_hat, Dt, stat_CM, stat_KS, booted_phi, booted_dt, booted_CM, booted_KS = out_fn(T, phi, num_boot, boot_fraction)

alphas = np.linspace(0, 1, 50)
Dt_q = np.quantile(booted_dt, alphas)
phi_hat_q = np.quantile(booted_phi, alphas)
CM_q = np.quantile(booted_CM, alphas)
KS_q = np.quantile(booted_KS, alphas)

sim_setting['result'] = dict(
    err=err, 
    phi_hat=phi_hat, 
    Dt=Dt, 
    CM=stat_CM,
    KS=stat_KS,
    quantiles_Dt=Dt_q, 
    quantiles_phi_hat=phi_hat_q,
    quantiles_CM=CM_q,
    quantiles_KS=KS_q)

sim_setting.to_pickle(out_name)