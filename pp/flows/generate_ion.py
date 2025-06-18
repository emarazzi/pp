from pymatgen.core import Structure
from dataclasses import dataclass, field
from jobflow import job, Flow, Maker
from atomate2.siesta.jobs.core import StaticMaker
from pathlib import Path
from glob import glob
import os
from pp.utils import cp_ion

@dataclass 
class GenerateIons(Maker):
    """
    Maker to generate the database of the ion files from siesta 
    from a specific set of pseudos


    """
    name = "Generate Ion Files"

    def make(
        self,
        pseudo_psml_dir: str | Path,
        save_ion_folder: str | Path,
        database_folder: str | Path,
    ) -> Flow:
        """
        Maker to generate all the siesta jobs, collect the ion files and 
        copy them in a save directory
        """

        jobs = []

        pseudos = glob(os.path.join(os.path.abspath(pseudo_psml_dir),'*psml'))
        database_folder = os.path.abspath(database_folder)

        for pseudo in pseudos:
            element  = pseudo.split('/')[-1].split('.')[0]
            if os.path.exists(os.path.join(database_folder,f"{element}.cif")):
                structure = Structure.from_file(os.path.join(database_folder,f'{element}.cif'))
                job_ion = StaticMaker().make(structure=structure)
                
                job_ion.name = element
                
                jobs.append(job_ion)
        job_copy = cp_ion(outdirs = [job.output for job in jobs],ion_dir = save_ion_folder)
        jobs.append(job_copy)

        return Flow(jobs=jobs,output = job_copy.output,name = self.name)
        