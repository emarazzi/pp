import numpy as np
from numpy import ndarray
from typing import Union, List
def band_comparison(band1: Union[List, ndarray], band2:Union[List, ndarray]):
    if len(band1[0]) != len(band2[0]):
        raise ValueError("bands do not have the same numbers of k points")
    band1 = np.array(band1)
    band2 = np.array(band2)
    sum = 0
    for b1,b2 in zip(band1,band2):
        sum += np.sum(abs(b1-b2)**2)
    sum /= band1.shape[1]+np.min([band1.shape[0],band2.shape[0]])
    
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