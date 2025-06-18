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
    #include_vacancies: bool = True
    ) -> list:

    structures_fname = []
    structure.make_supercell(supercell_size)
    j = 0
    while len(structures_fname) < size:
        j += 1
        structure = structure.perturb(distance=distance,min_distance=min_distance)
        fname = os.path.join(structures_dir,f'{j}.cif')
        structures_fname.append(fname)
        structure.to(fname)
    
    
    return structures_fname

