from pymatgen.core import Structure
from dataclasses import dataclass, field
from jobflow import Maker, job, Flow, Job
from pathlib import Path
from pp.jobs.mod_structure import generate_training_population
from pp.jobs.jobs import QEscf, QEpw2bgw, QEpw2bgw
from pp.jobs.hpro import HPROWrapper

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
    ion_dir: str | Path = './'

    def make(
        self,
        structure: Structure,
        ):
        """
        Create the flow to generate DFT data
        
        """

        jobs = []

        gen_structures_job = generate_training_population(
            structure = structure,
            structures_dir=self.structures_dir,
            distance = self.distance, 
            supercell_size = self.supercell_size,
            min_distance = self.min_distance, 
            size =  self.training_size, 
            #include_vacancies =  self.include_vacancies
            )
        jobs.append(gen_structures_job)
        


        qe_run_jobs = QEscf(    
            qe_run_cmd = self.qe_run_cmd,
            num_qe_workers = self.num_qe_workers,
            fname_pwi_template = self.fname_pwi_template,
            fname_structures = gen_structures_job.output
        )
        
        jobs.append(qe_run_jobs)

        pw2bgw_run_jobs = QEpw2bgw(
            scf_outdir = qe_run_jobs.output,
            name = 'Pw2Bgw Labelling',
            pw2bgw_command=self.pw2bgw_command,
            fname_pw2bgw_template=self.fname_pw2bgw_template,
            num_workers = 1
        )
        jobs.append(pw2bgw_run_jobs)

        hpro_job = HPROWrapper(
            qe_run_output = qe_run_jobs.output,
            ion_dir = self.ion_dir,
            ao_hamiltonian_dir = self.ao_hamiltonian_dir,
            upf_dir = self.upf_dir,
            ecutwfn = self.ecutwfn,
            metadata = {'pw2bgw_completed':pw2bgw_run_jobs.output}
        )

        jobs.append(hpro_job)

        return Flow(jobs,output=[j.output for j in jobs],name=self.name)
