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
    def __init__(self, HSPoints: List, avecs: np.ndarray, ndivsm: int = 10):
        """
        HSPoints : list of fractional coordinates of high-symmetry points
        avecs    : 3x3 array of real-space lattice vectors as rows (in Å)
        ndivsm   : number of divisions for the shortest segment (QE style)
        """
        self.HSPoints = np.array(HSPoints)
        self.avecs = np.array(avecs)
        self.ndivsm = ndivsm

        # Compute reciprocal lattice vectors: b_i = 2π (A^{-1})^T
        self.bvecs = 2 * np.pi * np.linalg.inv(self.avecs).T

    def get_divisions(self):
        """
        Compute the number of divisions per segment
        so that the shortest segment has ndivsm divisions.
        """
        seglen = []
        for i in range(len(self.HSPoints) - 1):
            df = self.HSPoints[i+1] - self.HSPoints[i]
            dk = df @ self.bvecs  # convert to reciprocal-space Cartesian coords
            seglen.append(np.linalg.norm(dk))
        seglen = np.array(seglen)

        minlen = np.min(seglen)
        divisions = seglen / minlen * self.ndivsm
        divisions = np.rint(divisions).astype(int)
        return divisions

    def print_qe_path(self, filename: str | None = None):
        """
        Print the QE-style path file section:
        number of points
        followed by kx ky kz ndiv
        """
        divisions = self.get_divisions()
        divisions = np.concatenate((divisions, [1]))  # QE convention

        lines = [f"{len(self.HSPoints)}"]
        for k, d in zip(self.HSPoints, divisions):
            lines.append(f"{k[0]:.6f}  {k[1]:.6f}  {k[2]:.6f}  {d}")

        text = "\n".join(lines)

        if filename:
            with open(filename, "a") as f:
                f.write(text + "\n")
        else:
            print(text)


def read_eig_hpro(filename: str):
    """
    Read eigenvalues from a file in the hpro format.
    """
    file = open(filename, "r")
    lines = file.readlines()
    nbnd = int(lines[2].split()[1])
    bands = [[] for _ in range(nbnd)]
    max_ov = 0
    for line in lines[3:]:
        nn = line.split()
        if len(nn) == 3:
            bands[int(line.split()[1])-1].append(float(line.split()[2]))
        elif len(nn) == 2:
            max_ov = np.max([max_ov,int(nn[1][0])])
    file.close()
    return np.array(bands[max_ov:])

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
        json.dump(d, json_file, indent=4)
    

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
