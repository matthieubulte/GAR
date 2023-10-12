import numpy as np

def comp_auto_dist(d):
    # Input:
    #   d: n by n distance matrix, the (i,j)-component denotes the distance d(yi,yj) between yi and yj
    # Output:
    #   Ax, Ay: U-centered versions of the auto-distance matrix
    n = d.shape[0]
    num_k = n - 4
    Ax = np.zeros((n, n, num_k))
    Ay = np.zeros((n, n, num_k))
    
    for k in range(num_k):
        dx = d[(k+1):n, (k+1):n]
        dy = d[:(n-k-1), :(n-k-1)]
        
        Ax[:(n-k-1), :(n-k-1), k] = dx - np.outer(np.sum(dx, axis=1) / (n-k-2), np.sum(dx, axis=0) / (n-k-2)) + np.sum(dx) / ((n-k-1) * (n-k-2))
        Ay[:(n-k-1), :(n-k-1), k] = dy - np.outer(np.sum(dy, axis=1) / (n-k-2), np.sum(dy, axis=0) / (n-k-2)) + np.sum(dy) / (((n-k-1)) * (n-k-2))
        
        np.fill_diagonal(Ax[:(n-k-1), :(n-k-1), k], 0)
        np.fill_diagonal(Ay[:(n-k-1), :(n-k-1), k], 0)

    return Ax, Ay

def prop_test(d, B=500):
    n = d.shape[0]
    num_k = n - 4
    zeta = np.arange(0, np.pi, 0.01)
    rep_CM = np.zeros(B)
    rep_KS = np.zeros(B)

    Ax, Ay = comp_auto_dist(d)
    
    Sn = np.sum(Ax * Ay, axis=(0, 1)) / (n - np.arange(1, num_k + 1) - 3) / (np.arange(1, num_k + 1) * np.pi)
    Sn = Sn @ np.sin(np.outer(np.arange(1, num_k + 1), zeta))

    stat_CM = np.mean(Sn**2) * np.pi
    stat_KS = np.max(np.abs(Sn))

    for b in range(B):
        order_b = np.random.permutation(n)
        Ax_b, Ay_b = comp_auto_dist(d[order_b, :][:, order_b])
        
        Sn_b = np.sum(Ax_b * Ay_b, axis=(0, 1)) / (n - np.arange(1, num_k + 1) - 3) / (np.arange(1, num_k + 1) * np.pi)
        Sn_b = Sn_b @ np.sin(np.outer(np.arange(1, num_k + 1), zeta))

        rep_CM[b] = np.mean(Sn_b**2) * np.pi
        rep_KS[b] = np.max(np.abs(Sn_b))

    return stat_CM, rep_CM, stat_KS, rep_KS


