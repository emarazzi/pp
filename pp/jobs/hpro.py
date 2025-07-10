from HPRO import PW2AOkernel
from HPRO.lcaodiag import LCAODiagKernel
from HPRO.structure import load_structure, save_pymatgen_structure
from jobflow import job, Flow, Response
from pathlib import Path
import os
from typing import Any, List, Union, Optional
import numpy as np

__all__ = [
    'HPROWrapper'
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
    
    Returns:
        Flow with the HPRO jobs    
    """
    
    output = {'ao_dirs':[]}
    jobs = []
    qe_output_folders = np.hstack([output['outdir'] for output in qe_run_output])
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
    kptwts: List,
    kptsymbol: List,
    efermi: Optional[float] = None
) -> Flow:
    output: dict = {'ao_dirs':[]}
    jobs: List = []
    for ao_dir in ao_hamiltonian_dir:
        hpro_job = diag(
            ao_dir = ao_dir,
            nbnd = nbnd,
            kpts = kpts,
            kptwts = kptwts,
            kptsymbol = kptsymbol,
            efermi = efermi
        )
        jobs.append(hpro_job)
        output['ao_dirs'].append(ao_dir)
    
    flow = Flow(jobs=jobs,output=output)

    return Response(replace=flow,output=output)


@job
def diag(
    ao_dir: Union[str, Path],
    nbnd: int,
    kpts: List,
    kptwts: List,
    kptsymbol: List,
    efermi: Optional[float] = None
) -> None:
    kernel = LCAODiagKernel()
    kernel.setk(
        kpts=kpts,
        kptwts=kptwts,
        kptsymbol=kptsymbol,
        )

    kernel.load_deeph_mats(ao_dir)
    kernel.diag(nbnd=nbnd, efermi=efermi)
    kernel.write(ao_dir)
    structure = load_structure(path=ao_dir,interface='deeph')
    save_pymatgen_structure(structure,os.path.join(ao_dir,'structure.cif'))


