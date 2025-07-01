from pymatgen.core import Structure
from dataclasses import dataclass, field
from jobflow import Flow, Maker, Job
from atomate2.siesta.jobs.core import StaticMaker
from atomate2.siesta.sets.core import StaticSetGenerator
from pathlib import Path
from glob import glob
from typing import Union, List
import os
from pp.utils import cp_ion

@dataclass
class GenerateIons(Maker):
    """
    Maker to generate the database of the ion files from siesta 
    from a specific set of pseudos

    Args:
        name: str
            Name of the Flow


    """
    name: str = "Generate Ion Files"

    def make(
        self,
        pseudo_psml_dir: Union[str, Path],
        save_ion_folder: Union[str, Path],
        database_folder: Union[str, Path],
        basis_set: str = field(default='DZ')
    ) -> Flow:
        """
        Generate all the SIESTA jobs, collect the ion files, and
        copy them to a save directory

        Args:
            pseudo_psml_dir: Directory containing pseudos files (.psml)
            save_ion_folder: Directory where ion files will be saved
            database_folder: Directory containing CIF files for structures

        Returns:
            A Flow
        """
        jobs: List[Job] = []

        # Find all pseudopotential files in the specified directory
        pseudos: List[str] = glob(os.path.join(os.path.abspath(pseudo_psml_dir),'*psml'))

        # Convert database_folder to abs path
        database_folder = os.path.abspath(database_folder)

        for pseudo in pseudos:
            # Extract the element name from the pseudopotential file name
            element: str = pseudo.split('/')[-1].split('.')[0]

            # Check if a corresponding CIF file exists for the element
            cif_path: str = os.path.join(database_folder, f"{element}.cif")
            if os.path.exists(cif_path):
                structure: Structure = Structure.from_file(cif_path)
                static_set_generator = StaticSetGenerator(basis_set=basis_set)
                job_ion: Job = StaticMaker(input_set_generator=static_set_generator).make(structure=structure)

                job_ion.name = element
                
                jobs.append(job_ion)

        # Create a job to copy the ion files to the save directory
        job_copy: Job = cp_ion(
            outdirs = [job.output for job in jobs],
            ion_dir = save_ion_folder
        )
        jobs.append(job_copy)

        return Flow(jobs=jobs,output = job_copy.output,name = self.name)

