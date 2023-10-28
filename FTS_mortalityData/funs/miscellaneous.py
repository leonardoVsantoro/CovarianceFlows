import numpy as np
from kneed import KneeLocator
from numpy.linalg import norm

# silouhette score funs
def cohesion(i, labels, X): 
    other_ixs = np.arange(X.shape[0]); other_ixs = other_ixs[other_ixs!=i]
    if len(other_ixs) > 0:
        coh =  np.array([norm(X[i]- X[j]) for j in other_ixs]).mean() 
    else:
        coh = 1
    return coh
def separation(i, labels, X): 
    other_labels = np.unique(labels); other_labels[other_labels!=labels[i]]
    sep = np.inf
    for lab in other_labels:
        ixs = labels[labels==lab]
        _ =  np.array([norm(X[i]- X[j]) for j in ixs]).mean() 
        if _< sep:
            sep = _
    return sep


# average consecutive time series
def calculate_moving_average(X, N):
    moving_averages = []
    for i in range(len(X) - N + 1):
        window = X[i:i + N]
        avg = sum(window) / N
        moving_averages.append(avg)
    return np.array(moving_averages)