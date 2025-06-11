from HPRO import PW2AOkernel
from jobflow import job, Flow, Response
from pathlib import Path
import os
from typing import Any
import numpy as np

__all__ = [
    'HPROWrapper'
]

@job
def HPROWrapper(
    qe_run_output: list,
    siesta_ouput: str,
    ao_hamiltonian_dir: str | Path,
    upf_dir: str | Path = os.getenv('ESPRESSO_PSEUDO','./'),
    ecutwfn: int | float = 30.0
) -> Flow:
    
    output = {'ao_dirs':[]}
    jobs = []
    qe_output_folders = np.hstack([output['outdir'] for output in qe_run_output])
    print(type(siesta_ouput))
    for j,qe_output_folder in enumerate(qe_output_folders):
        os.makedirs(os.path.join(ao_hamiltonian_dir,str(j)),exist_ok=True)
        hpro_job = ReconstructWrapper(
            qe_folder = qe_output_folder,
            siesta_path = siesta_ouput,
            ao_hamiltonian_dir = os.path.join(ao_hamiltonian_dir,str(j)),
            upf_dir = upf_dir,
            ecutwfn = ecutwfn,
        )
        jobs.append(hpro_job)
        output['ao_dirs'].append(os.path.join(ao_hamiltonian_dir,str(j)))
    
    flow = Flow(jobs=jobs,output=output)

    return Response(replace=flow)


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