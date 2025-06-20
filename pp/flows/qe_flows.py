from pp.jobs.labelling import QEscf, QEnscf, QEband
from dataclasses import dataclass, field
from jobflow import Maker, Flow

@dataclass
class ElectronBS(Maker):
    """
    Maker for generating electron band structure data using Quantum ESPRESSO.
    """
    name: str = "ElectronBS"
    qe_run_cmd: str = "mpirun -np 1 pw.x"
    num_qe_workers: int | None = None
    fname_scf_template: str | None = None
    fname_nscf_template: str | None = None
    fname_bands_template: str | None = None

    def make(self,structure_file: str) -> Flow:
        """
        Create the flow to generate electron band structure data.
        """

        jobs = []

        # Create the job for static calculation
        scf_job = QEscf(
            name="Static Calculation",
            qe_run_cmd=self.qe_run_cmd,
            num_qe_workers=self.num_qe_workers,
            fname_pwi_template=self.fname_scf_template,
            fname_structures=structure_file
        )

        jobs.append(scf_job)

        # Create the job for nscf calculation
        nscf_job = QEnscf(
            name="NSCF Calculation",
            qe_run_cmd=self.qe_run_cmd,
            num_qe_workers=self.num_qe_workers,
            fname_pwi_template=self.fname_nscf_template,
            scf_outdir=scf_job.output            
        )

        jobs.append(nscf_job)

        # Create the job for band structure calculation
        band_job = QEband(
            name="Band Structure Calculation",
            qe_run_cmd=self.qe_run_cmd,
            num_qe_workers=self.num_qe_workers,
            fname_pwi_template=self.fname_bands_template,
            fname_structures=structure_file
        )

        jobs.append(band_job)

        flow = Flow(jobs=jobs, output=[j.output for j in jobs], name=self.name)


        return flow




