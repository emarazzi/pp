from pymatgen.core import Structure
from dataclasses import dataclass, field
from jobflow import Maker, job, Flow, Job
from pathlib import Path
from pp.jobs.jobs import QEscf, QEpw2bgw
from pp.jobs.hpro import HPROWrapper
from typing import List, Tuple, Union, Optional
import os
from pp.utils import generate_training_population
__all__ = [
    'GenerateDFTData'
]

@dataclass
class GeneratePopulationSettings:
    """
    Class to handle settings related to generate_training_population job.
    
    Args:
    -----
    run_generate_population:
      Whether to run the run_generate_population
    structures_dir: Union[str, Path]
      Directory to store the structures cif files
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
      Size of the training dataset (Default: 500)
    include_vacancies: bool
      Whether to include structure with vacancies in your structure database.
      Still not implemented.
    """
    structures_dir: Union[str, Path] = './'
    distance: float = 0.02
    min_distance: Optional[float] = 0.0001
    supercell_size: Union[List[int], Tuple[int, ...]] = field(default_factory=lambda: [1, 1, 1])
    training_size: int = 500
    include_vacancies: bool = False

@dataclass
class QEScfSettings:
    """
    Class to handle settings related to QE scf
    
    Args:
    -----
    run_qe_scf:
      Whether to run scf calculation
    structures_names:
      list of names of paths to structures to run QE
    qe_run_cmd: str
      Command to execute pw.x
    fname_pwi_template: str
      Path to the template for pw scf calculations
    kspace_resolution: Optional[float] = None
      K-space resolution for the scf calculations in Angstrom^-1.
    koffset: list[bool] = field(default_factory=lambda: [False, False, False])
      K-point offset for the scf calculations.
    num_qe_workers: Optional[int] = None
      Number of workers to execute pw.x calculations.
      Default to None that corresponds to one worker per structure
    enforce_2d: bool
      Whether to enforce 2D k-point sampling (i.e., set k-point in z direction to 1).
    """
    structures_names: Optional[List[str]] = None
    qe_run_cmd: str = "srun --mpi=cray_shasta $PATHQE/bin/pw.x"
    fname_pwi_template: str = "scf.in"
    kspace_resolution: Optional[float] = None
    koffset: List[bool] = field(default_factory=lambda: [False, False, False])
    num_qe_workers: Optional[int] = None
    enforce_2d: bool = False

@dataclass
class QEPw2bgwSettings:
    """
    Class to handle settings related to QE pw2bgw
    
    Args:
    -----
    run_pw2bgw:
      Whether to run pw2bgw calculations
    qe_scf_outdir:
      list of dict with qe output paths and success status
      must contain 'outdir' and 'success' keys.
    pw2bgw_run_cmd: str
      As qe_run_cmd but for pw2bgw.x
    fname_pw2bgw_template: str
      As fname_pwi_template but for pw2bgw.x
    num_p2b_workers: Optional[int]
      As for num_qe_workers but for pw2bgw.x
    """
    qe_scf_outdir: Optional[List[dict]] = None
    pw2bgw_run_cmd: str = "$PATHQE/bin/pw2bgw.x"
    fname_pw2bgw_template: str = "pw2bgw.in"
    num_p2b_workers: Optional[int] = None

@dataclass
class HPROSettings:
    """
    Class to handle settings related to HPRO
    
    Args:
    -----
    ao_hamiltonian_dir: Union[str, Path]
      Directory to store the preprocessed data from HPRO
    upf_dir: Union[str, Path]                                                                       
      Directory containing the upf pseudos for QE.
      Default is the environment variable $ESPRESSO_PSEUDO
    ecutwfn: Union[int, float]
      ecutwfn variable from QE, used only for HPRO.
      To set a specific value of ecutwfn in your QE calculation
      modify the scf template.  
    ion_dir: Union[str, Path]
      Directory containing the ion files generated with siesta
    qe_scf_outdir:
      list of dict with qe output paths and success status
      must contain 'outdir' and 'success' keys.
      If only HPRO is run, folders must contain both prefix.save and VSC files
    """
    ao_hamiltonian_dir: Union[str, Path] = './'  
    upf_dir: Union[str, Path] = os.getenv('ESPRESSO_PSEUDO', './')
    ecutwfn: Union[int, float] = 30.0   
    ion_dir: Union[str, Path] = './'
    qe_scf_outdir: Optional[List[dict]] = None

@dataclass
class GenerateDFTData(Maker):
    """
    A Maker to generate the training dataset from dft PW computations
    and siesta ion files

    Args:
    ----
    name: str
      Name of the Flow
    population_settings: Optional[GeneratePopulationSettings]
      Class to handle settings related to generate_training_population job.
    scf_settings: Optional[QEScfSettings]
      Class to handle settings related to QE scf
    pw2bgw_settings: Optional[QEPw2bgwSettings]
      Class to handle settings related to QE pw2bgw
    hpro_settings: Optional[HPROSettings]
      Class to handle settings related to HPRO
    """
    name: str = "GenerateDFTData"
    # Settings
    population_settings: Optional[GeneratePopulationSettings] = field(default_factory=GeneratePopulationSettings)
    scf_settings: Optional[QEScfSettings] = field(default_factory=QEScfSettings)
    pw2bgw_settings: Optional[QEPw2bgwSettings] = field(default_factory=QEPw2bgwSettings)
    hpro_settings: Optional[HPROSettings] = field(default_factory=HPROSettings)
    

    def __post_init__(self):
        if self.scf_settings is not None and not (self.scf_settings.structures_names or self.population_settings is not None):
            raise ValueError("You should either run the generate_training_population job \
                              or provide a list of structures names to run QE.")
        if self.pw2bgw_settings is not None and not (self.pw2bgw_settings.qe_scf_outdir is not None or self.scf_settings is not None):
            raise ValueError("You should either run the QEscf job \
                              or provide a list of dict with qe output paths and success status to run pw2bgw.")
        if self.hpro_settings is not None and not (self.hpro_settings.qe_scf_outdir or self.pw2bgw_settings):
            raise ValueError("You should either run the QEpw2bgw job \
                              or provide a list of dict with qe output paths and success status \
                              that contains the VSC files to run HPRO.")
        
    def make(
        self,
        structure: Optional[Structure] = None,
    ) -> Flow:
        """
        Create the flow to generate the training dataset.

        Args:
          structure (Optional[Structure]): The input structure. If not provided,
            the generate_training_population job will fail unless pre-existing
            structures are specified in the SCF settings.

        Returns:
          Flow: A JobFlow Flow that represents the end-to-end dataset generation.
        """

        jobs: List[Job] = []

        # Initialize job references to None
        gen_structures_job = None
        qe_run_jobs = None
        pw2bgw_run_jobs = None

        if self.population_settings:
            structures_dir = Path(self.population_settings.structures_dir)
            if not structures_dir.exists():
                structures_dir.mkdir(parents=True, exist_ok=True)
            gen_structures_job = generate_training_population(
                structure=structure,
                structures_dir=self.population_settings.structures_dir,
                distance=self.population_settings.distance,
                supercell_size=self.population_settings.supercell_size,
                min_distance=self.population_settings.min_distance,
                size=self.population_settings.training_size,
            )
            jobs.append(gen_structures_job)

        if self.scf_settings:
            qe_run_jobs = QEscf(dict(
                qe_run_cmd=self.scf_settings.qe_run_cmd,
                num_qe_workers=self.scf_settings.num_qe_workers,
                fname_pwi_template=self.scf_settings.fname_pwi_template,
                fname_structures=gen_structures_job.output if gen_structures_job else self.scf_settings.structures_names,
                kspace_resolution=self.scf_settings.kspace_resolution,
                koffset=self.scf_settings.koffset,
                enforce_2d=self.scf_settings.enforce_2d,
            ))
            jobs.append(qe_run_jobs)

        if self.pw2bgw_settings:
            pw2bgw_run_jobs = QEpw2bgw(
                scf_outdir=qe_run_jobs.output if qe_run_jobs else self.pw2bgw_settings.qe_scf_outdir,
                pw2bgw_command=self.pw2bgw_settings.pw2bgw_run_cmd,
                fname_pw2bgw_template=self.pw2bgw_settings.fname_pw2bgw_template,
                num_workers=self.pw2bgw_settings.num_p2b_workers
            )
            jobs.append(pw2bgw_run_jobs)

        if self.hpro_settings:
            ao_dir = Path(self.hpro_settings.ao_hamiltonian_dir)
            if not ao_dir.exists():
                ao_dir.mkdir(parents=True, exist_ok=True)
            hpro_job = HPROWrapper(
                qe_run_output=qe_run_jobs.output if qe_run_jobs else self.hpro_settings.qe_scf_outdir,
                ion_dir=self.hpro_settings.ion_dir,
                ao_hamiltonian_dir=self.hpro_settings.ao_hamiltonian_dir,
                upf_dir=self.hpro_settings.upf_dir,
                ecutwfn=self.hpro_settings.ecutwfn,
                metadata={'has_pw2bgw_completed': pw2bgw_run_jobs.output if pw2bgw_run_jobs else False},
            )

            jobs.append(hpro_job)

        return Flow(jobs, output=[j.output for j in jobs], name=self.name)
