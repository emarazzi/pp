from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from typing import List
import os
import json
from pathlib import Path
from shutil import copy
import re
from jobflow import job

def generate_training_population(
    structure: Structure,
    structures_dir: str,
    supercell_size: List[int] = [1, 1, 1],
    distance: float = 0.1,
    min_distance: float | None = None,
    size: int = 200,
) -> List[str]:
    """
    Generate a set of perturbed structures starting from a reference Pymatgen Structure.

    This function takes an input Pymatgen Structure, expands it into a supercell,
    and generates multiple perturbed copies by randomly displacing atomic positions.
    Each perturbed structure is written to disk as a CIF file in the specified directory.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The input structure to use as a reference. This object will not be modified.
    structures_dir : str
        Path to the directory where generated CIF files will be saved.
        The directory will be created if it does not exist.
    supercell_size : List[int], optional
        Supercell expansion factors along each lattice vector, e.g. [2, 2, 2].
        Default is [1, 1, 1].
    distance : float, optional
        Maximum displacement magnitude (in Å) applied to atomic positions
        during perturbation. Default is 0.1 Å.
    min_distance : float, optional
        Minimum displacement magnitude (in Å) applied to atomic positions
        during perturbation. Default is None, i.e. everything is perturb by same amount.
    size : int, optional
        Number of perturbed structures to generate. Default is 200.

    Returns
    -------
    List[str]
        List of full file paths (CIF format) corresponding to the generated perturbed structures.

    """
    # Ensure output directory exists
    os.makedirs(structures_dir, exist_ok=True)

    # Create a supercell reference structure
    base_structure = structure.copy()
    base_structure.make_supercell(supercell_size)

    structures_fname: List[str] = []

    for j in range(size):
        # Work with a fresh copy each time to avoid cumulative perturbations
        perturbed = base_structure.copy()
        perturbed.perturb(distance=distance, min_distance=min_distance)

        fname = os.path.join(structures_dir, f"{j}.cif")
        perturbed.to( filename=fname)
        structures_fname.append(fname)

        # Free memory from perturbed structure explicitly (optional for large loops)
        del perturbed

    return structures_fname


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
