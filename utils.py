from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from typing import List


def standard_primitive(file_in,file_out=None):
    structure = Structure.from_file(file_in)
    structure_primitive = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
    if file_out:
        structure_primitive.to(file_out)
    else:
        structure_primitive.to(f"{file_in.split('.')[0]}-prim.cif")
    return structure_primitive



class KPath:
    def __init__(self,HSPoints: List,ndivsm:int=10):
        self.HSPoints = HSPoints
        self.ndivsm = ndivsm
    def get_weights(self):
        dist = np.array([np.linalg.norm(np.array(self.HSPoints[i])-np.array(self.HSPoints[i+1])) for i in range(len(self.HSPoints)-1)])
        mindist = min(dist)
        weights = dist / mindist * self.ndivsm
        return weights
    def print_qe_path(self,filename:str=None):
        weights = self.get_weights()
        weights = np.concatenate((weights,np.array([1])))
        if filename:
            with open(filename,"w") as file:
                file.write(f"{len(self.HSPoints)}\n")
                for w,k in zip(weights,self.HSPoints):
                    file.write(f"{k[0]:.6f}  {k[1]:.6f}  {k[2]:.6f}  {w:.0f}\n")
        else:
            print(f"{len(self.HSPoints)}")
            for w,k in zip(weights,self.HSPoints):
                print(f"{k[0]:.6f}  {k[1]:.6f}  {k[2]:.6f}  {w:.0f}")
    def print_openmx_path(self,filename:str=None):
        weights = self.get_weights()
        if filename:
            with open(filename,"w") as file:
                for i,w in enumerate(weights):
                    file.write(f"{w:.0f}  {self.HSPoints[i][0]:.6f}  {self.HSPoints[i][1]:.6f}  {self.HSPoints[i][2]:.6f}\
   {self.HSPoints[i+1][0]:.6f}  {self.HSPoints[i+1][1]:.6f}  {self.HSPoints[i+1][2]:.6f}\n")
        else:
            for i,w in enumerate(weights):
                print(f"{w:.0f}  {self.HSPoints[i][0]:.6f}  {self.HSPoints[i][1]:.6f}  {self.HSPoints[i][2]:.6f}\
   {self.HSPoints[i+1][0]:.6f}  {self.HSPoints[i+1][1]:.6f}  {self.HSPoints[i+1][2]:.6f}")


def read_eig_hpro(filename: str):
    """
    Read eigenvalues from a file in the hpro format.
    """
    file = open(filename, "r")
    lines = file.readlines()
    nbnd = int(lines[2].split()[1])
    bands = [[] for _ in range(nbnd)]
    for line in lines[3:]:
        nn = line.split()
        if len(nn) == 3:
            bands[int(line.split()[1])-1].append(float(line.split()[2]))
    file.close()
    return bands

