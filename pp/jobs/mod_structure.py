from pymatgen.core import Structure
from random import randint, choice
from jobflow import job
import os
from pathlib import Path
from typing import List, Tuple, Union, Optional
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
    structures_dir: Union[str, Path],
    supercell_size: Union[List, Tuple] = [1,1,1],
    distance: float = 0.1, 
    min_distance: Optional[Union[float,int]] = 0.0,  
    size: int = 200, 
    include_vacancies: bool = True
    ) -> list:
    """
    Job to generate N structure in the desidered supercell with the desired supercell
    Based on pymatgen Structure.perturb

    Args:
    -----
    structure: Structure
        an input pymatgen structure
    structures_dir: Union[str, Path]
        folder to save the generated structures
    supercell_size: Union[List, Tuple]
        size of the supercell
    distance: float
        distance in angstroms by which to perturb each site.
    min_distance: Optional[Union[float,int]]
        if None, all displacements will be equal amplitude. 
        If int or float, perturb each site a
        distance drawn from the uniform distribution between
        'min_distance' and 'distance'.
    size: int
        size of the population
    include_vacances: bool
        whether to include structures with vacancies
        Not impletemented yet
    
    Returns:
        list with the path to the generated structures
    
    """

    structures_fname = []
    structure.make_supercell(supercell_size)
    j = 0
    while len(structures_fname) < size:
        j += 1
        stru = structure.perturb(distance=distance,min_distance=min_distance)
        fname = os.path.join(structures_dir,f'{j}.cif')
        structures_fname.append(fname)
        stru.to(fname)
    
    
    return structures_fname

