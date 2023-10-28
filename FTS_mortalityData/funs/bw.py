# BW funs 
import numpy as np
from numpy.linalg import pinv,inv,norm,matrix_rank
from scipy.linalg import sqrtm 



def optmap(F,G, reg=0):
    Id = np.diag(np.ones(F.shape[0]))
    F = F+reg*(Id+1j*Id)
    G = G+reg*(Id+1j*Id)
    sqrtm_F = sqrtm(F).astype('complex128')
    try:
        sqrtm_inv_F = inv(sqrtm_F)
    except:
        sqrtm_inv_F = pinv(sqrtm_F)
        
    T = sqrtm_inv_F@sqrtm(sqrtm_F@G@sqrtm_F)@sqrtm_inv_F
    return T

def BW(F,G):
    sqrtm_F = sqrtm(F)
    Id = np.diag(np.ones(F.shape[0]))
    return (trace(F) + trace(G) - 2*trace(sqrtm(sqrtm_F@G@sqrtm_F).real))**.5

def FM(Fs, MaxIter=2000, tol =1e-3, reg=1e-3):
    __check = None
    Id=np.diag(np.ones(Fs.shape[1]))
    bary = Fs.mean(0)
    
    Fs = np.array([F+ Id*reg for F in Fs])
    Fs = np.array([F+ 1j*Id*reg for F in Fs])
        
    n = len(Fs)
    
    for k in range(MaxIter): 
        T_k = np.array([optmap(bary, F) for F in Fs]).mean(0)
        if norm(T_k - Id) < tol:
            __check = True
            break
        else:
            bary = T_k @ bary @ T_k
        if k > 10 and norm(T_k - Id) > 5*1e2:
            __check = False;
            break
    __norm = norm(T_k - Id)
    return bary, __check, __norm

