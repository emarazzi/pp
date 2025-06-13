from pymatgen.core import Structure
from random import randint, choice
from jobflow import job
import os
from pathlib import Path
from typing import List, Tuple
def remove_atom(structure:Structure):
    """
    Remove a random atom from the structure.
    The modified structure is returned, the modification does not happen in place.
    """
    idx = randint(0, len(structure) - 1)
    structure = structure.copy()
    structure.remove_sites([idx])
    return structure

@job
def generate_training_population(
    structure:Structure,
    structures_dir: str | Path,
    supercell_size: List | Tuple = [1,1,1],
    distance: float = 0.1, 
    min_distance: float | None = 0.0,  
    size: int = 200, 
    include_vacancies: bool = True
    ) -> list:

    structures = []
    structures_fname = []
    
    j = 0
    fname = os.path.join(structures_dir,f'{j}.cif')
    structures_fname.append(fname)
    structure.make_supercell(supercell_size)
    structure.to(fname)
    
    j += 1
    structure_vac = remove_atom(structure) if include_vacancies else structure.perturb(distance=distance,min_distance=min_distance)
    fname = os.path.join(structures_dir,f'{j}.cif')
    structures_fname.append(fname)
    structure_vac.to(fname)
    structures = [structure,structure_vac]

    while j < size-1:
        j += 1
        structure = choice(structures).perturb(distance=distance,min_distance=min_distance)
        fname = os.path.join(structures_dir,f'{j}.cif')
        structures_fname.append(fname)
        structure_vac.to(fname)
    
    
    return structures_fname



@job
def add(a,b):
    return a+b