from HPRO import PW2AOkernel
from HPRO.lcaodiag import LCAODiagKernel
from HPRO.structure import load_structure, save_pymatgen_structure
from pp.utils import KPath
from jobflow import job, Flow, Response, Job
from pathlib import Path
import os
from typing import Any, List, Union, Optional
import numpy as np

__all__ = [
    'HPROWrapper',
    'ReconstructWrapper',
    'DiagWrapper',
    'diag'
]

@job
def HPROWrapper(
    qe_run_output: List,
    ion_dir: Union[str, Path],
    ao_hamiltonian_dir: Union[str, Path],
    upf_dir: Union[str, Path] = os.getenv('ESPRESSO_PSEUDO','./'),
    ecutwfn: Union[int, float] = 30.0,
    metadata: dict[str, Any] = {}
) -> Flow:
    """
    Wrapper to prepare the HPRO jobs

    Args:
    ----
    qe_run_output: List[dict]
        list of dictionaries with the outputs of the QE workers
    ion_dir: Union[str, Path]
        folder with the x.ion files
    ao_hamiltonian_dir: Union[str, Path]
        folder to save the generated data
    upf_dir: Union[str, Path]
        folder with upf pseudos for QE
        Default is the $ESPRESSO_PSEUDO environment variable
    ecutwfn:
        Energy cutoff in Hartree
    metadata: dict
        Dummy dictionary to add optional dependencies
    
    Returns:
        Flow with the HPRO jobs    
    """
    
    output:dict = {'ao_dirs':[]}
    jobs: List[Job] = []
    if np.all([isinstance(out, dict) for out in qe_run_output]):
        success = np.hstack([out['success'] for out in qe_run_output]).tolist()
        scf_outdir = np.hstack([out['outdir'] for out in qe_run_output]).tolist()
        if not np.all(success):
            print("Not all scf calculation where successful, only the successfull one will be parsed to HPRO.")
        qe_output_folders = [dir for dir,succ in zip(scf_outdir,success) if succ]
        
    for j,qe_output_folder in enumerate(qe_output_folders):
        folder_name = qe_output_folder.split('/')[-1]
        os.makedirs(os.path.join(ao_hamiltonian_dir,folder_name),exist_ok=True)
        hpro_job = ReconstructWrapper(
            qe_folder = qe_output_folder,
            siesta_path = ion_dir,
            ao_hamiltonian_dir = os.path.join(ao_hamiltonian_dir,folder_name),
            upf_dir = upf_dir,
            ecutwfn = ecutwfn,
        )
        jobs.append(hpro_job)
        output['ao_dirs'].append(os.path.join(ao_hamiltonian_dir,folder_name))
    
    flow = Flow(jobs=jobs,output=output)

    return Response(replace=flow,output=output)


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
    qe_folder:
        path to the QE output dir containing the VSC file from pw2bgw.x
    siesta_path:
        path to the folder with x.ion files
    ao_hamiltonian_dir:
        folder to save the generated data
    upf_dir:
        path to upf pseudos.
        Default is the environmental variable ESPRESSO_PSEUDO
    ecutwfn
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


@job
def DiagWrapper(    
    ao_hamiltonian_dir: List,
    nbnd: int,
    kpts: List,
    kptsymbol: List,
    kptwts: Optional[List] = None,
    ndivsm: Optional[int] = None,
    hmatfname: str = 'hamiltonians.h5',
    efermi: Optional[float] = None
) -> Flow:
    """
    Wrapper to prepare the diagonalization job, using diag from HPRO

    Args:
    -----
    ao_hamiltonian_dir:
        list of dirs containing data to diagonalize.
        In each folder there must be the structure in deeph format, 
        and the overlaps.h5 and hamiltonians.h5 files
    nbnd:
        number of bands
    kpts:
        high symmetry path
    kptwts:
        weights for the high symmetry path
    ndivsm:
        number of division in the shortest segment
    kptsymbol:
        symbols for the high symmetry path
    hmatfname:
        name of the file containing the hamiltonian to diagonalize
    efermi:
        value of the Fermi level in eV

    Returns:
    --------
        Flow with the diag jobs        
    
    """
    output: dict = {'ao_dirs':[]}
    jobs: List = []
    if kptwts is None and ndivsm is None:
        raise ValueError("One of kptwts or ndivsm must be not None.")
    elif kptwts is not None and ndivsm is not None:
        UserWarning("Both kptwts and ndivsm are not None, kptwts is used and ndivsm discarded.")
    elif kptwts is None and ndivsm is not None:
        kpoints = KPath(HSPoints=kpts,ndivsm=ndivsm)
        weights = kpoints.get_weights().tolist()
        kptwts = weights + [1]
        
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

    return Response(replace=flow,output=output)


@job
def diag(
    ao_dir: Union[str, Path],
    nbnd: int,
    kpts: List,
    kptwts: List,
    kptsymbol: List,
    eigfname: str = 'eig.dat',
    hmatfname: str = 'hamiltonians.h5',
    efermi: Optional[float] = None
) -> None:
    """
    Job to diagonalize the hamiltonian

    Args:
    -----
    ao_dir:
        dir containing data to diagonalize.
        In each folder there must be the structure in deeph format, 
        and the overlaps.h5 and hamiltonians.h5 files
    nbnd:
        number of bands
    kpts:
        high symmetry path
    kptwts:
        weights for the high symmetry path
    kptsymbol:
        symbols for the high symmetry path
    hmatfname:
        name of the file containing the hamiltonian to diagonalize
    efermi:
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


