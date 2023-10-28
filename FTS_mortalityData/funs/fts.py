# library 
import numpy as np

# Freq Dom funs
def f(r_t, omega):
    return 1/(2*np.pi) * np.array( [np.exp(- 1j*omega*t)*__r_t for t,__r_t in enumerate(r_t)]).mean(0)

def r(X, t, ix, iy):
    T = X.shape[0]
#     X = np.array([x-X.mean(0) for x in X])
    return (X[t:,ix]@X[:T-t,iy])/(T-t) - X[t:,ix].mean()*X[:T-t,iy].mean()