"""
Jobs for running the workflow.
This is a copy from 41bY/src/autoplex/auto/GenMLFF/jobs.py
credits to Alberto Pacini
"""

from dataclasses import field
from jobflow import job, Flow, Response
from pp.jobs.labelling import QEstaticLabelling, QEpw2bgwLabelling , QEnscfLabelling, QEbandLabelling, qe_params_from_config
import numpy as np
from typing import List, Union, Optional, Dict
from pp.utils import KPath

@job
def QEscf(
    params: dict | None = None,
):
    """
    Initialize the QEScfLabelling with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the QEScfLabelling.
    """
    #Initialize QEstaticLabelling with the provided parameters
    qe_params = qe_params_from_config(params)    

    # Execute QE static labelling
    # return a list containing (list[success], list[pwo], list[outdir]) for each worker
    output_per_worker = QEstaticLabelling(**qe_params).make()

    return output_per_worker

@job
def QEnscf(
    scf_outdir: list[str] | list[dict], #Path to the output directory of the scf calculation
    name: str = "NSCF labelling",
    qe_run_cmd: str = "mpirun -np 1 pw.x",
    num_qe_workers: int | None = 1, #Number of workers to use for the calculations. If None setp up 1 worker per scf
    fname_pwi_template: str | None = None, #Path to file containing the template QE input
    
):
    """
    Initialize the QEScfLabelling with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the QEScfLabelling.
    """
    if np.all([isinstance(out, dict) for out in scf_outdir]):
        success = np.hstack([out['success'] for out in scf_outdir]).tolist()
        scf_outdir = np.hstack([out['outdir'] for out in scf_outdir]).tolist()
        if not np.all(success):
            print("Not all scf calculation where successful, only the successfull one will be used.")
        scf_outdir = [dir for dir,succ in zip(scf_outdir,success) if succ]
    #Parameters for QENScfLabelling
    qe_params = {
        "name": name,
        "qe_run_cmd": qe_run_cmd,
        "num_qe_workers": num_qe_workers,
        "fname_pwi_template": fname_pwi_template,
        "scf_outdir": scf_outdir,
    }

    # Execute QE static labelling
    # and return the paths to the labelled structures
    output_per_worker = QEnscfLabelling(**qe_params).make()

    return output_per_worker

@job
def QEband(
    scf_outdir: list[str] | list[dict],
    name: str = "Band structure labelling",
    qe_run_cmd: str = "mpirun -np 1 bands.x",
    num_qe_workers: int | None = 1, #Number of workers to use for the calculations. If None setp up 1 worker per scf
    fname_pwi_template: str | None = None, #Path to file containing the template QE input
    kpath: list | None = None, # high symmetry path
    ndivsm: int | None = None  # number of points in the shortest high symmetry segment
):
    """
    Initialize the QEScfLabelling with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the QEScfLabelling.
    """
    if np.all([isinstance(out, dict) for out in scf_outdir]):
        success = np.hstack([out['success'] for out in scf_outdir]).tolist()
        scf_outdir = np.hstack([out['outdir'] for out in scf_outdir]).tolist()
        if not np.all(success):
            print("Not all scf calculation where successful, only the successfull one will be used.")
        scf_outdir = [dir for dir,succ in zip(scf_outdir,success) if succ]
    if kpath and ndivsm:
        kpoints = KPath(HSPoints=kpath,ndivsm=ndivsm)
    qe_params = {
        "name": name,
        "qe_run_cmd": qe_run_cmd,
        "num_qe_workers": num_qe_workers,
        "fname_pwi_template": fname_pwi_template,
        "scf_outdir": scf_outdir,
        "kpoints": kpoints
    }

    output_per_worker = QEbandLabelling(**qe_params).make()

    return output_per_worker

@job
def QEpw2bgw(
    scf_outdir: List[str] | List[dict],
    name: str = 'pw2bgw',
    pw2bgw_command : str = 'pw2bgw.x -in',
    fname_pw2bgw_template: str | None = None,
    num_workers: int | None = None
):
    """
    Initialize the QEpw2bgwLabelling with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the QEpw2bgwLabelling.
    """
    if np.all([isinstance(out, dict) for out in scf_outdir]):
        success = np.hstack([out['success'] for out in scf_outdir]).tolist()
        scf_outdir = np.hstack([out['outdir'] for out in scf_outdir]).tolist()
        if not np.all(success):
            print("Not all scf calculation where successful, only the successfull one will be parsed to HPRO.")
        scf_outdir = [dir for dir,succ in zip(scf_outdir,success) if succ]
    
    pw2bgw_params = {
        'name': name,
        'pw2bgw_command': pw2bgw_command,
        'fname_pw2bgw_template': fname_pw2bgw_template,
        'scf_outdir': scf_outdir,
        'num_workers': num_workers
    }    
    output_per_worker = QEpw2bgwLabelling(**pw2bgw_params).make()

    return output_per_worker

