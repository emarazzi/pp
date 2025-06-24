from pymatgen.core import Structure
from dataclasses import dataclass, field
from jobflow import Maker, job, Flow, Job
from pathlib import Path
from pp.jobs.mod_structure import generate_training_population
from pp.jobs.jobs import QEscf, QEpw2bgw
from pp.jobs.hpro import HPROWrapper
from typing import List, Tuple, Union, Optional
import os

__all__ = [
    'GenerateDFTData'
]

@dataclass
class GenerateDFTData(Maker):
    """
    A Maker to generate the training dataset from dft PW computations
    and siesta ion files

    Args:
        name: str
              Name of the flows
        structures_dir: Union[str, Path]
              Directory to store the structures cif files
        ao_hamiltonian_dir: Union[str, Path]
              Directory to store the preprocessed data from HPRO
        distance: float
              Distance in angstroms by which to perturb each site.
        min_distance: float
              if None, all displacements will be equal amplitude. 
              If int or float, perturb each site a distance drawn 
              from the uniform distribution between 'min_distance' and 'distance'.
        supercell_size: Union[List[int], Tuple[int, ...]]
              A scaling matrix for transforming the lattice
              vectors. Has to be all integers
        training_size: int
              Size of the training dataset (Default:500)
        include_vacances: bool
              Whether to include structure with vacances in your structure database.
              Still not implemented.
        qe_run_cmd: str
              Command to execute pw.x
        fname_pwi_template: str
              Path to the template for pw scf calculations
        upf_dir: Union[str, Path]                                                                       
              Directory containing the upf pseudos for QE.
              Default is the environment variable $ESPRESSO_PSEUDO
        ecutwfn: Union[int, float]
              ecutwfn variable from QE, used only for HPRO.
              To set a specific value of ecutwfn in your QE calculation
              modify the scf template.
        kspace_resolution: Optional[float] = None
                  K-space resolution for the scf calculations in Angostrom^-1.
        koffset: list[bool] = field(default_factory=lambda: [False, False, False])
                  K-point offset for the scf calculations.
        num_qe_workers: Optional[int] = None
              Number of workers to execute pw.x calculations.
              Default to None that corresponds to one worker per structure
        pw2bgw_run_cmd: str
              As qe_run_cmd but for pw2bgw.x
        fname_pw2bgw_template: str
              As fname_pwi_template but for pw2bgw.x
        ion_dir: Union[str, Path]
              Directory containing the ion files generated with siesta
    """
    name: str = "GenerateDFTData"   
    structures_dir: Union[str, Path] = './'
    ao_hamiltonian_dir: Union[str, Path] = './'
    distance: float = 0.1
    min_distance: Optional[float] = 0.001
    supercell_size: Union[List[int], Tuple[int, ...]] = field(default_factory=lambda: [1, 1, 1])
    training_size: int = 500
    include_vacancies: bool = False

    qe_run_cmd: str = "srun --mpi=cray_shasta $PATHQE/bin/pw.x"
    fname_pwi_template: str = "scf.in"
    upf_dir: Union[str, Path] = os.getenv('ESPRESSO_PSEUDO', './')
    ecutwfn: Union[int, float] = 30.0
    kspace_resolution: Optional[float] = None
    koffset: List[bool] = field(default_factory=lambda: [False, False,False])
    num_qe_workers: Optional[int] = None
    pw2bgw_run_cmd: str = "$PATHQE/bin/pw2bgw.x"
    fname_pw2bgw_template: str = "pw2bgw.in"
    ion_dir: Union[str, Path] = './'

    def make(
        self,
        structure: Structure,
    ) -> Flow:
        """
        Create the flow to generate the training dataset.

        Args:
            structure: Structure
                     The input structure

        Returns:
            A Flow
        """
        jobs: List[Job] = []

        gen_structures_job = generate_training_population(
            structure = structure,
            structures_dir = self.structures_dir,
            distance = self.distance, 
            supercell_size = self.supercell_size,
            min_distance = self.min_distance, 
            size =  self.training_size,     
        )
        jobs.append(gen_structures_job)

        qe_run_jobs = QEscf(dict(
            qe_run_cmd = self.qe_run_cmd,
            num_qe_workers = self.num_qe_workers,
            fname_pwi_template = self.fname_pwi_template,
            fname_structures = gen_structures_job.output
        ))
        jobs.append(qe_run_jobs)

        pw2bgw_run_jobs = QEpw2bgw(
            scf_outdir = qe_run_jobs.output,
            name = 'Pw2Bgw Labelling',
            pw2bgw_command = self.pw2bgw_run_cmd,
            fname_pw2bgw_template = self.fname_pw2bgw_template,
            num_workers = 1
        )
        jobs.append(pw2bgw_run_jobs)

        hpro_job = HPROWrapper(
            qe_run_output = pw2bgw_run_jobs.output,
            ion_dir = self.ion_dir,
            ao_hamiltonian_dir = self.ao_hamiltonian_dir,
            upf_dir = self.upf_dir,
            ecutwfn = self.ecutwfn,
        )

        jobs.append(hpro_job)

        return Flow(jobs, output = [j.output for j in jobs], name=self.name)
