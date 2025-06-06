"""
Jobs for running the workflow.
This is a copy from 41bY/src/autoplex/auto/GenMLFF/jobs.py
credits to Alberto Pacini
"""

from dataclasses import field
from jobflow import job, Flow, Response
from pp.dft_calc.labelling import QEstaticLabelling
@job
def QEscf(
    name: str = "do_qe_static_labelling",
    qe_run_cmd: str = "mpirun -np 1 pw.x",
    num_qe_workers: int | None = 1, #Number of workers to use for the calculations. If None setp up 1 worker per scf
    fname_pwi_template: str | None = None, #Path to file containing the template QE input
    fname_structures: str | None = None, #Path to ASE-readible file containing the structures to be computed
):
    """
    Initialize the QEScfLabelling with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the QEScfLabelling.
    """
    #Parameters for QEScfLabelling
    qe_params = {
        "name": name,
        "qe_run_cmd": qe_run_cmd,
        "num_qe_workers": num_qe_workers,
        "fname_pwi_template": fname_pwi_template,
        "fname_structures": fname_structures,
    }

    # Execute QE static labelling
    # and return the paths to the labelled structures
    dict_of_fout_and_success = QEstaticLabelling(**qe_params).make()

    return dict_of_fout_and_success

