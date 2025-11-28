from HPRO import PW2AOkernel
from HPRO.lcaodiag import LCAODiagKernel
from HPRO.structure import load_structure, save_pymatgen_structure
from pp.utils import KPath
from jobflow import job, Flow, Response, Job
from pathlib import Path
from typing import Any
import os
import numpy as np

__all__ = [
    'HPROWrapper',
    'ReconstructWrapper',
    'DiagWrapper',
    'diag'
]

@job
def HPROWrapper(
    qe_run_output: list[dict],
    ion_dir: str | Path,
    ao_hamiltonian_dir: str | Path,
    upf_dir: str | Path = os.getenv('ESPRESSO_PSEUDO','./'),
    ecutwfn: int | float = 30.0,
    metadata: dict[str, Any] = {}
) -> Flow:
    """
    Wrapper to prepare the HPRO jobs

    Args:
    ----
    qe_run_output: list[dict]
        list of dictionaries with the outputs of the QE workers
    ion_dir: str | Path
        folder with the x.ion files
    ao_hamiltonian_dir: str | Path
        folder to save the generated data
    upf_dir: str | Path
        folder with upf pseudos for QE
        Default is the $ESPRESSO_PSEUDO environment variable
    ecutwfn: int | float
        Energy cutoff in Hartree
    metadata: dict[str, Any]
        Dummy dictionary to add optional dependencies
    
    Returns:
        Flow with the HPRO jobs    
    """
    
    outputs:dict = {'ao_dirs':[]}
    jobs: list[Job] = []
    if np.all([isinstance(out, dict) for out in qe_run_output]):
        # extract only successful folders
        success = np.hstack([out['success'] for out in qe_run_output]).tolist()
        scf_outdir = np.hstack([out['outdir'] for out in qe_run_output]).tolist()
        if not np.all(success):
            print("Not all scf calculation where successful, only the successfull one will be parsed to HPRO.")
        qe_output_folders = [dir for dir,succ in zip(scf_outdir,success) if succ]
        
    for j,qe_output_folder in enumerate(qe_output_folders):
        folder_name = qe_output_folder.split('/')[-1]
        folder = Path(ao_hamiltonian_dir)/folder_name
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        hpro_job = ReconstructWrapper(
            qe_folder = qe_output_folder,
            siesta_path = ion_dir,
            ao_hamiltonian_dir = folder,
            upf_dir = upf_dir,
            ecutwfn = ecutwfn,
        )
        jobs.append(hpro_job)
        outputs['ao_dirs'].append(hpro_job.output)
    
    flow = Flow(jobs=jobs,output=outputs)

    return Response(replace=flow,output=flow.output)


@job
def ReconstructWrapper(
    qe_folder: str | Path = './',
    siesta_path : str | Path = './',
    ao_hamiltonian_dir: str | Path = './',
    upf_dir: str | Path = os.getenv('ESPRESSO_PSEUDO','./'),
    ecutwfn: int | float = 30.0
) -> None:
    """
    Projection job

    Args:
    ----
    qe_folder: str | Path
        path to the QE output dir containing the VSC file from pw2bgw.x
    siesta_path: str | Path
        path to the folder with x.ion files
    ao_hamiltonian_dir: str | Path
        folder to save the generated data
    upf_dir: str | Path
        path to upf pseudos.
        Default is the environmental variable ESPRESSO_PSEUDO
    ecutwfn: int | float
        energy cutoff used in qe calculation
    
    """
    kernel = PW2AOkernel(
        lcao_interface='siesta',
        lcaodata_root=siesta_path,
        hrdata_interface='qe-bgw',
        vscdir=os.path.join(qe_folder,'VSC'),
        upfdir=upf_dir,
        ecutwfn=ecutwfn
    )
    kernel.run_pw2ao_rs(ao_hamiltonian_dir)

    return ao_hamiltonian_dir

@job
def DiagWrapper(    
    ao_hamiltonian_dir: list[str | Path],
    nbnd: int,
    kpts: list,
    kptsymbol: list,
    kptwts: list | None = None,
    ndivsm: int | None = None,
    avecs: list | None = None,
    hmatfname: str = 'hamiltonians.h5',
    efermi: float | None = None
) -> Flow:
    """
    Wrapper to prepare the diagonalization job, using diag from HPRO

    Args:
    -----
    ao_hamiltonian_dir: list[str | Path]
        list of dirs containing data to diagonalize.
        In each folder there must be the structure in deeph format, 
        and the overlaps.h5 and hamiltonians.h5 files
    nbnd: int
        number of bands
    kpts: list
        high symmetry path
    kptwts: list | None
        weights for the high symmetry path
    ndivsm: int | None
        number of division in the shortest segment
    avecs: list | None
        lattice vectors, needed only if kptwts is None
    kptsymbol: list
        symbols for the high symmetry path
    hmatfname: str
        name of the file containing the hamiltonian to diagonalize
    efermi: float | None
        value of the Fermi level in eV

    Returns:
    --------
        Flow with the diag jobs        
    
    """
    output: dict = {'ao_dirs':[]}
    jobs: list = []
    if kptwts is None and ndivsm is None:
        raise ValueError("One of kptwts or ndivsm must be not None.")
    elif kptwts is not None and ndivsm is not None:
        UserWarning("Both kptwts and ndivsm are not None, kptwts is used and ndivsm discarded.")
    elif kptwts is None and ndivsm is not None:
        if avecs is None:
            raise ValueError("If kptwts is None, avecs must be provided to compute the weights.")
        kpoints = KPath(HSPoints=kpts,avecs=avecs,ndivsm=ndivsm)
        divisions = kpoints.get_divisions().tolist()
        kptwts = divisions + [1]
        
    for ao_dir in ao_hamiltonian_dir:
        name = ao_dir.split('/')[-1]
        hpro_job = diag(
            ao_dir = ao_dir,
            nbnd = nbnd,
            kpts = kpts,
            kptwts = kptwts,
            kptsymbol = kptsymbol,
            eigfname = f"{name}.dat",
            hmatfname = hmatfname,
            efermi = efermi
        )
        jobs.append(hpro_job)
        output['ao_dirs'].append(ao_dir)
        structure = load_structure(path=ao_dir,interface='deeph')
        save_pymatgen_structure(structure,os.path.join(ao_dir,f'{name}.cif'))
    
    flow = Flow(jobs=jobs,output=output)

    return Response(replace=flow,output=flow.output)


@job
def diag(
    ao_dir: str | Path,
    nbnd: int,
    kpts: list,
    kptwts: list,
    kptsymbol: list,
    eigfname: str = 'eig.dat',
    hmatfname: str = 'hamiltonians.h5',
    efermi: float | None = None
) -> str | Path:
    """
    Job to diagonalize the hamiltonian

    Args:
    -----
    ao_dir: str | Path
        dir containing data to diagonalize.
        In each folder there must be the structure in deeph format, 
        and the overlaps.h5 and hamiltonians.h5 files
    nbnd: int
        number of bands
    kpts: list
        high symmetry path
    kptwts: list
        weights for the high symmetry path
    kptsymbol: list
        symbols for the high symmetry path
    hmatfname: str = 'hamiltonians.h5'
        name of the file containing the hamiltonian to diagonalize
    efermi: float | None = None
        value of the Fermi level in eV
    """
    kernel = LCAODiagKernel()
    kernel.setk(
        kpts=kpts,
        kptwts=kptwts,
        kptsymbol=kptsymbol,
        )

    kernel.load_deeph_mats(ao_dir, hmatfname=hmatfname)
    kernel.diag(nbnd=nbnd, efermi=efermi)
    kernel.write(ao_dir,eigfname=eigfname)

    return ao_dir


