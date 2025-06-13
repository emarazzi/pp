from pymatgen.core import Structure
from dataclasses import dataclass, field
from jobflow import Maker, job, Flow, Job
from pathlib import Path
from pp.mod_structure import generate_training_population
from pp.dft_calc.jobs import QEscf, QEpw2bgw, WrapperQEpw2bgw
from atomate2.siesta.jobs.core import StaticMaker
from pp.hpro import HPROWrapper
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from typing import List, Tuple, Union


import os

__all__ = [
    'GenerateDFTData'
]

@dataclass
class GenerateDFTData(Maker):
    """
    
    """
    name: str = "GenerateDFTData"
    
    structures_dir: str | Path = './'
    ao_hamiltonian_dir: str | Path = './'
    distance: float = 0.1
    supercell_size: List | Tuple = field(default_factory=lambda: [1, 1, 1])
    min_distance: float | None = None
    training_size: int = 500
    include_vacancies: bool = False

    qe_run_cmd: str = "srun --mpi=cray_shasta $PATHQE/bin/pw.x"
    fname_pwi_template: str = "scf.in"
    upf_dir: str | Path = os.getenv('ESPRESSO_PSEUDO','./')
    ecutwfn: int | float = 30.0
    num_qe_workers: int | None = None
    pw2bgw_command: str = "srun --mpi=cray_shasta $PATHQE/bin/pw2bgw.x"
    fname_pw2bgw_template: str = "pw2bgw.in"

    def make(
        self,
        structure: Structure,
        ao_structure: Structure
        ):
        """
        Create the flow to generate DFT data
        
        """

        jobs = []
        
        siesta_job = StaticMaker().make(structure=SpacegroupAnalyzer(ao_structure).get_primitive_standard_structure())
        
        jobs.append(siesta_job)

        gen_structures_job = generate_training_population(
            structure = structure,
            structures_dir=self.structures_dir,
            distance = self.distance, 
            supercell_size = self.supercell_size,
            min_distance = self.min_distance, 
            size =  self.training_size, 
            include_vacancies =  self.include_vacancies
            )
        jobs.append(gen_structures_job)
        


        qe_run_jobs = QEscf(    
            qe_run_cmd = self.qe_run_cmd,
            num_qe_workers = self.num_qe_workers,
            fname_pwi_template = self.fname_pwi_template,
            fname_structures = gen_structures_job.output
        )
        
        jobs.append(qe_run_jobs)

        pw2bgw_run_jobs = WrapperQEpw2bgw(
            name = 'Pw2Bgw Labelling',
            pw2bgw_command=self.pw2bgw_command,
            fname_pw2bgw_template=self.fname_pw2bgw_template,
            qe_output = qe_run_jobs.output,
            num_workers = 1
        )
        jobs.append(pw2bgw_run_jobs)

        hpro_job = HPROWrapper(
            qe_run_output = qe_run_jobs.output,
            siesta_ouput = siesta_job.output,
            ao_hamiltonian_dir = self.ao_hamiltonian_dir,
            upf_dir = self.upf_dir,
            ecutwfn = self.ecutwfn,
            metadata = {'pw2bgw_completed':pw2bgw_run_jobs.output}
        )

        jobs.append(hpro_job)

        return Flow(jobs,output=[j.output for j in jobs],name=self.name)
