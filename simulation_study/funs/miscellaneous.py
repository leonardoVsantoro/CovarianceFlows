import numpy as np
from sklearn.gaussian_process.kernels import Matern


def BM_cov(d): 
    grid = np.linspace(0,1,d)
    return np.array([[min(s,t) for s in grid] for t in grid])
def BB_cov(d):
    grid = np.linspace(0,1,d)
    return np.array([[min(s,t) - s*t for s in grid] for t in grid])

def matern_cov(d,nu): 
    matern_grid = np.array([ (
            np.linspace(0,1,d).ravel()[:, np.newaxis], 
            np.linspace(0,1,d).ravel()[:, np.newaxis]) ])[0].T.reshape(-1,2)
    return Matern(nu=nu)(matern_grid)