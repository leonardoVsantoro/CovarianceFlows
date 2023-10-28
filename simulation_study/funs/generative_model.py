import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy
from scipy.integrate import simps as simps
from random import randrange
from scipy.interpolate import interp1d
from numpy.random import normal,uniform,chisquare,binomial
from scipy.stats import chi2, wishart
from numpy.random import multivariate_normal
from math import sin,cos
from scipy import signal
from scipy.fftpack import fft,rfft, irfft, ifft, dct, idct, dst, idst, fftshift, fftfreq
from numpy import pi 
from numpy.linalg import pinv,inv,norm,matrix_rank
from scipy.linalg import sqrtm 

def generate_point(mean_x, mean_y, deviation_x, deviation_y, mean_z =None, deviation_z =None):
    if mean_z is None:
        return np.random.normal(mean_x, deviation_x), np.random.normal(mean_y, deviation_y)
    else:
        return np.random.normal(mean_x, deviation_x), np.random.normal(mean_y, deviation_y),  
    np.random.normal(mean_z, deviation_z)


def random_curve_LAMBDA(N, std=2):
    time_grid = np.linspace(0,1,N)
    eps = 1e-1; ker = (signal.windows.hann(N)**(1/eps))[np.arange(-N//2,N//2)]; ker = ker/ker.sum()
    
    nn =  randrange(4) +2; ixs = list(set( [0] + [ randrange(N) for _ in range(nn) ] + [N-1])); ixs.sort()
    ts = np.array([time_grid[ix] for ix in ixs]) ; 
    Y = np.array([ uniform(0.1,2.1) for t in ts])
    Y_out =  np.real(ifft( fft(interp1d(ts, Y, kind='linear')(time_grid))*fft(ker) ))
    Y_out = Y_out - Y_out.mean()+1
    return Y_out

def random_curve_THETA( N):
    time_grid = np.linspace(0,1,N)
    eps = 1e-1; ker = (signal.windows.hann(N)**(1/eps))[np.arange(-N//2,N//2)]; ker = ker/ker.sum()
        
    nn =  randrange(4) +2; ixs = list(set( [0] + [ randrange(N) for _ in range(nn) ] + [N-1])); ixs.sort()
    ts = np.array([time_grid[ix] for ix in ixs]) ; Y = np.array([ uniform(-pi,pi) for t in ts])
    Y_out =  np.real(ifft( fft(interp1d(ts, Y, kind='linear')(time_grid))*fft(ker) ))
    return Y_out
    
def get_bias(cutoff,d):
    Id = np.diag(np.ones(d))
    Dgrid = np.linspace(0,1,d)
    PSIS = np.array( [ np.sin(_n*2*np.pi * (Dgrid) ).reshape(-1,1) for _n in np.arange(1,cutoff)] 
                    + [ np.cos(_n*2*np.pi * (Dgrid)).reshape(-1,1) for _n in np.arange(1,cutoff)])
    T = (1/cutoff) * np.array([  _PSI @_PSI.T for  _PSI in  PSIS ]).sum(0)
    T = sqrtm(T  + 1e-5*Id).real; 
    return  (T@T.T)- Id

def FreqDom_GenModel(d,cutoff, theta =None, lambdas =None, k=5,mean =1):
    Id = np.diag(np.ones(d))
    Dgrid = np.linspace(0,1,d)
    if theta is None:
         theta =  uniform(-pi,pi)
            
    PSIS = np.array( [ np.sin(_n*2*np.pi * (Dgrid -theta) ).reshape(-1,1)
                          for _n in np.arange(1,cutoff)] +
                [ np.cos(_n*2*np.pi * (Dgrid-theta)).reshape(-1,1)
                          for _n in np.arange(1,cutoff)])
    if lambdas is None:
        lambdas =  np.array([(1/k * chisquare(k)) for _ in PSIS])
    T = (1/cutoff) * np.array([ _lambda * _PSI @_PSI.T for (_lambda, _PSI) in zip(lambdas, PSIS)]).sum(0)

    T = sqrtm(T  + 1e-5*Id).real; 
    return  (T@T.T)-get_bias(cutoff,d)
    
def FreqDom_GenModel_FLOW(N,d, cutoff, theta_flow = None, lambdas_flow = None, k=5, meanFlow=1, LR=None, plot=False):
    time_grid = np.linspace(0,1,N)
    if theta_flow is None:
         theta_flow= random_curve_THETA(N)
    if lambdas_flow is None:
        lambdas_flow= random_curve_LAMBDA(N);
    lambdas_flow = lambdas_flow*meanFlow
    if LR is not None:
        lambdas =  np.array([uniform(LR[0],LR[1]) for _ in range(d)])
    else:
        lambdas =  np.array([(1/k * chisquare(k)) for _ in range(d)])
    T = np.array([FreqDom_GenModel(d,cutoff, theta = theta_,lambdas=_lambdas*lambdas, k=k) for theta_, _lambdas in zip(theta_flow,lambdas_flow)])

    if plot:
        fig,[axL,axR] = plt.subplots(figsize = (10,2),ncols=2)
        axL.plot(time_grid, theta_flow); axR.plot(time_grid, lambdas_flow);
        axL.set_title('$\Theta$'); axR.set_title('$\lambda$')
        axL.set_ylim(-pi,pi)
        plt.show()
        vmin = T.min(); vmax = T.max()
        fig,axs = plt.subplots(figsize = (18,3), ncols = 11)
        for j, ax in enumerate(axs):
            closeix = np.argmin((time_grid-j/10)**2)
            sns.heatmap(T[closeix], square = True, vmin = vmin, vmax = vmax, cbar=False, ax=ax)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_xlabel('{:.1f}'.format(j/10))
#         plt.show()
    return T

