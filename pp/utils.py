from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from typing import List
import os
import json
from jobflow import job
from pathlib import Path
from shutil import copy
import re
from glob import glob

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

def write_dh_structure(structure:Structure, save_dir: str = './'):
    with open(os.path.join(save_dir,'element.dat'),'w') as file:
        for species in structure.species:
            file.write(f"{species.symbol}\n")
    lattice = np.transpose(structure.lattice.matrix)
    with open(os.path.join(save_dir,'lat.dat'),'w') as file:
        for vec in lattice:
            file.write(f"{vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}\n")
    cart_coords = np.transpose(structure.cart_coords)
    with open(os.path.join(save_dir,'site_positions.dat'),'w') as file:
        for coords in cart_coords:
            file.write(f"{coords[0]:.10f}  {coords[1]:.10f}  {coords[2]:.10f}\n")
    d = {"isspinful": False, "fermi_level": 0.0}
    with open(os.path.join(save_dir,"info.json"), "w") as json_file:
        json.dump(d, json_file, indent=4)  # `indent` makes the output readable
    

@job
def cp_ion(outdirs: str | Path | list[str] | list[Path], ion_dir: str | Path) -> Path:
    element = re.compile(r'\w+\.')
    if isinstance(outdirs, str | Path):
        outdirs = [Path(outdirs)]
    else:
        outdirs = [Path(outdir) for outdir in outdirs]
    ion_dir = Path(ion_dir)
    for outdir in outdirs:
        ion_files = list(outdir.glob('*ion'))
        ion_file = [ionf for ionf in ion_files if len(element.findall(ionf.name))==1]
        for ionf in ion_file:
            copy(src=ionf,dst=ion_dir)

    return ion_dir
