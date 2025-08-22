import numpy as np
from numpy import ndarray
from typing import Union, List
def band_comparison(band1: Union[List, ndarray], band2:Union[List, ndarray]) -> float:
    if len(band1[0]) != len(band2[0]):
        raise ValueError("bands do not have the same numbers of k points")
    band1 = np.array(band1)
    band2 = np.array(band2)
    sum = 0
    for b1,b2 in zip(band1,band2):
        sum += np.sum(abs(b1-b2)**2)
    sum /= (band1.shape[1]+np.min([band1.shape[0],band2.shape[0]]))
    
    return np.sqrt(sum)

def shift_vbm(bands: Union[List, ndarray],fermie: float) -> ndarray:
    bands = np.array(bands)
    top_vb = np.max(bands[bands<fermie])
    bands -= top_vb
    return bands

def shift_cbm(bands: Union[List, ndarray],fermie: float) -> ndarray:
    bands = np.array(bands)
    bottom_cb = np.min(bands[bands>fermie])
    bands -= bottom_cb
    return bands

def get_bandgap(bands:Union[List, ndarray],fermie: float) -> float:
    bands = np.array(bands)
    top_vb = np.max(bands[bands<fermie])
    bottom_cb = np.min(bands[bands>fermie])
    return np.abs(bottom_cb-top_vb)

def fd(e, fermie, nu, sigma):
    return 1/(np.exp((e-fermie-nu)/sigma)+1)

def fdtilde(ref, new, fermie, nu, sigma):
    return np.sqrt(fd(ref,fermie,nu,sigma)*fd(new,fermie,nu,sigma))

def band_distance(ref, new, fermie, nu, sigma):
    if ref.shape[1] != new.shape[1]:
        raise RuntimeError("bands with different numbers of k points!")
    eta = 0
    for i in range(min(ref.shape[0],new.shape[0])):
       fdtilde_i = fdtilde(ref[i],new[i],fermie,nu,sigma)
       eta += np.sqrt(np.sum(fdtilde_i*(ref[i]-new[i])**2)/np.sum(fdtilde_i))
    return eta
