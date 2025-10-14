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

