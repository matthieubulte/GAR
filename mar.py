import numpy as np
from scipy.optimize import minimize
from pyfrechet.metric_spaces import MetricData

def phi_hat(M, x, mu_hat, tol=None):
    T = x.shape[0]
    tol = tol or 1.0 / np.sqrt(T)
    x = MetricData(M, x)
    def L(phi): 
        return np.array([ 
            M._d(x[j+1], M.geodesic(mu_hat, x[j], phi))**2 
            for j in range(T-1) 
        ]).mean()
    return minimize(L, np.random.rand(), method='Nelder-Mead', bounds=[(0,1)], options=dict(xatol=tol))['x'][0]

def bootstrap_phi_hat(M, x, B, s, phi_tol=None):
    bootstrap = np.zeros(B)
    S = int(np.floor(x.shape[0]) * s)
    for b in range(B):
        permed = np.random.permutation(x)
        mu_hat = MetricData(M, permed)[:S].frechet_mean()
        bootstrap[b] = phi_hat(M, permed, mu_hat, tol=phi_tol)
    return np.sort(bootstrap)

def estimate_mean_var_Dt(M, x, B):
    # mean: E[d(x1, x2)^2]
    # var: Var(d(x1, x2)^2) + 2 Cov(d(x1, x2)^2, d(x1, x3)^2)
    T = x.shape[0]
    D_XY_XZ = np.zeros((2, B))
    for b in range(B):
        idx = np.random.randint(T, size=3)
        _x = M.index(x, idx[0])
        _y = M.index(x, idx[1])
        _z = M.index(x, idx[2])

        D_XY_XZ[0, b] = M._d(_x, _y)**2
        D_XY_XZ[1, b] = M._d(_x, _z)**2

    mean_est = D_XY_XZ[0,:].mean()
    var_est = 