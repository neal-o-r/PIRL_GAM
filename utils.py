import numpy as np

'''
helper functions
'''

def svd_whiten(X):

        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X_white = np.dot(U, Vt)

        return X_white


def sigmoid(x): 
        return 1 / (1 + np.exp(-x))

def B_spline(x, n=10, k=3):
        '''
        x = Array of x points, 
        n = number of (equally spaced) knots
        k = spline degree
        '''
        if x.ndim == 1:
                x = x.reshape(1,-1)

        xl = np.min(x)
        xr = np.max(x)
        dx = (xr - xl) / n # knot seperation

        t = (xl + dx * np.arange(-k, n+1)).reshape(1,-1)
        T = np.ones_like(x).T * t # knot matrix
        X = x.T * np.ones_like(t) # x matrix
        P = (X - T) / dx # seperation in natural units
        B = ((T <= X) & (X < T+dx)).astype(int) # knot adjacency matrix
        r = np.roll(np.arange(0, t.shape[1]), -1) # knot adjacency mask

        # compute recurrence relation k times
        for ki in range(1, k+1):
                B = (P * B + (ki + 1 - P) * B[:, r]) / ki

        return B


def safe_reciprocal(a):
        """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
        with np.errstate(divide='ignore', invalid='ignore'):
                c = np.true_divide(1, a)
                c[~np.isfinite(c)] = 0  # -inf inf NaN
        return c

