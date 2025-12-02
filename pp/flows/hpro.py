
from dataclasses import dataclass, field
from monty import os
from jobflow import Maker, job, Flow, Job
from pathlib import Path
from pp.jobs.hpro import HPROWrapper, DiagWrapper

__all__ = [
    'HproFlowMaker'
]



@dataclass
class HproFlowMaker(Maker):
    """
    Maker to create a flow to project Hamiltonians using HPRO and diagonalize them.

    Args:
    ----
    qe_run_output: list[dict]
       list of dictionaries with the outputs of the QE workers
    ion_dir: str | Path
        folder with the x.ion files
    nbnd: int
        number of bands to consider
    kpts: list
        list of high-symmetry k-points
    kptsymbol: list
        list of high-symmetry k-point symbols
    ao_hamiltonian_dir: str | Path
        folder to use to store the projected Hamiltonians in AO basis
    upf_dir: str | Path = os.getenv('ESPRESSO_PSEUDO','./')
        path to upf pseudos
        Default is the environmental variable ESPRESSO_PSEUDO
    ecutwfn: int | float = 30.0
        energy cutoff used in qe calculation
    kptwts: list | None = None
        weights for the k-points
    ndivsm: int | None = None
        number of divisions in the shortest segment between two k-points
    avecs: list | None = None
        lattice vectors in Angstrom, needed only if kptwts is None
    hmatfname: str = 'hamiltonians.h5'
        name of the hamiltonian file
    efermi: float | None = None
        Fermi level to use in the diagonalization
    
    """


    qe_run_output: list[dict]
    ion_dir: str | Path
    nbnd: int
    kpts: list
    kptsymbol: list
    ao_hamiltonian_dir: str | Path
    upf_dir: str | Path = os.getenv('ESPRESSO_PSEUDO','./')
    ecutwfn: int | float = 30.0
    kptwts: list | None = None
    ndivsm: int | None = None
    avecs: list | None = None
    hmatfname: str = 'hamiltonians.h5'
    efermi: float | None = None

    def make(self) -> Flow:

        projecton_job = HPROWrapper(
            qe_run_output=self.qe_run_output,
            ion_dir=self.ion_dir,
            ao_hamiltonian_dir=self.ao_hamiltonian_dir,
            upf_dir=self.upf_dir,
            ecutwfn=self.ecutwfn
        )

        diag_job = DiagWrapper(
            ao_hamiltonian_dir=projecton_job.output,
            nbnd=self.nbnd,
            kpts=self.kpts,
            kptsymbol=self.kptsymbol,
            kptwts=self.kptwts,
            ndivsm=self.ndivsm,
            avecs=self.avecs,
            hmatfname=self.hmatfname,
            efermi=self.efermi
        )

        return Flow(jobs=[projecton_job, diag_job], output=diag_job.output)
