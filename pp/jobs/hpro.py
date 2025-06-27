from HPRO import PW2AOkernel
from jobflow import job, Flow, Response
from pathlib import Path
import os
from typing import Any, List, Union
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
        Default is to read the $ESPRESSO_PSEUDO environment variable
    ecutwfn:
        Energy cutoff in Hartree
    
    Returns:
        Flow with the HPRO jobs    
    """
    
    output = {'ao_dirs':[]}
    jobs = []
    qe_output_folders = np.hstack([output['outdir'] for output in qe_run_output])
    
    for j,qe_output_folder in enumerate(qe_output_folders):
        os.makedirs(os.path.join(ao_hamiltonian_dir,str(j)),exist_ok=True)
        hpro_job = ReconstructWrapper(
            qe_folder = qe_output_folder,
            siesta_path = ion_dir,
            ao_hamiltonian_dir = os.path.join(ao_hamiltonian_dir,str(j)),
            upf_dir = upf_dir,
            ecutwfn = ecutwfn,
        )
        jobs.append(hpro_job)
        output['ao_dirs'].append(os.path.join(ao_hamiltonian_dir,str(j)))
    
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