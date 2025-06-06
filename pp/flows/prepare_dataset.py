from pymatgen.core import Structure
from dataclasses import dataclass
from jobflow import Maker, job, Flow, Job
from pathlib import Path
from pp.mod_structure import generate_training_population
from pp.dft_calc.jobs import QEscf
from atomate2.siesta.jobs.core import StaticMaker
from pp.hpro import HPROWrapper
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
    supercell_size: list| tuple = [1,1,1]
    min_distance: float | None = None
    training_size: int = 500
    include_vacancies: bool = False

    qe_run_cmd: str = "srun --mpi=cray_shasta $PATHQE/bin/pw.x"
    fname_pwi_template: str = "scf.in"
    upf_dir: str | Path = os.getenv('ESPRESSO_PSEUDO','./')
    ecutwfn: int | float = 30.0

    @job
    def maker(
        self,
        structure: Structure,
        ao_structure: Structure
        ):
        """
        Create the flow to generate DFT data
        
        """

        jobs: list[Job] = []

        gen_structures_job = generate_training_population(
            structure = structure,
            distance = self.distance, 
            supercell_size = self.supercell_size,
            min_distance = self.min_distance, 
            size =  self.training_size, 
            include_vacancies =  self.include_vacancies
            )
        jobs.append(gen_structures_job)
        
        qe_run_jobs = QEscf(    
            qe_run_cmd = self.qe_run_cmd,
            num_qe_workers = 1,
            fname_pwi_template = self.fname_pwi_template,
            fname_structures = gen_structures_job.output
        )
        
        jobs.append(qe_run_jobs)
        
        siesta_job = StaticMaker().make(structure=ao_structure)
        
        jobs.append(siesta_job)

        hpro_job = HPROWrapper(
            qe_run_output = qe_run_jobs.output,
            siesta_ouput = siesta_job.output,
            ao_hamiltonian_dir = self.ao_hamiltonian_dir,
            upf_dir = self.upf_dir,
            ecutwfn = self.ecutwfn
        )

        jobs.append(hpro_job)

        return Flow(jobs,output=[j.output for j in jobs],name=self.name)






        






        