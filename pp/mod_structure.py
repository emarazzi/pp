from pymatgen.core import Structure
from random import randint,random
import numpy as np

def remove_atom(structure:Structure):
    """
    Remove a random atom from the structure.
    The modified structure is returned, the modification does not happen in place.
    """
    idx = randint(0, len(structure) - 1)
    structure = structure.copy()
    structure.remove_sites([idx])
    return structure

def move_atom(structure:Structure, displacement:float=0.1):
    """
    Move a random atom by a displacement in a random direction.
    The modified structure is returned, the modification does not happen in place.
    """
    structure = structure.copy()
    idx = randint(0, len(structure) - 1)
    direction = [0,0,0]
    while np.sum(direction) == 0:
        direction = [randint(-1, 1) for _ in range(3)]
    norm = sum(d**2 for d in direction)**0.5
    direction = np.array([d / norm * displacement for d in direction])
    coords = structure.cart_coords
    new_coords = [coords[i]+ direction if i == idx else coords[i] for i in range(len(coords))]

    return Structure(structure.lattice, structure.species, new_coords, coords_are_cartesian=True, to_unit_cell=True)

def mod_structure(structure:Structure, nmod:int=1, displacement:float=0.1,move_bias:float=0.8):
    """
    Modify a structure by removing or moving atoms.
    The modified structure is returned, the modification does not happen in place.
    """
    for _ in range(nmod):
        if random() > move_bias:
            structure = remove_atom(structure=structure)
        else:
            structure = move_atom(structure=structure, displacement=displacement)
    return structure

def generate_training_population(structure:Structure, distance:float=0.1, min_distance:float|None=0.0, seed:int|None=None, size:int=200, include_vacancies:bool=True) -> Structure:
    structures = [structure]
    for j in range(size-1):
        if j == 0 and include_vacancies:
            structures.append(remove_atom(structure))
        else:
            structures.append(structure.perturb(distance=distance,min_distance=min_distance,seed=seed))
    return structures



