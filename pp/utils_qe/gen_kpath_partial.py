import numpy as np


nk = 10e6

istart = 2400000

iend = 2600000


kptfull = np.linspace(0,1,int(nk),endpoint=False)
weight = 1/nk

with open('pathk.dat','w') as file:
    file.write(f'{iend-istart} crystal\n')
    for k in kptfull[istart:iend]:
        file.write(f"0.00000000   0.00000000   {k:.8f}   {weight:.8f}\n")

