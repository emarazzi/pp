from pp.jobs.jobs import QEscf, QEband, QEnscf
from dataclasses import dataclass, field
from jobflow import Maker, Flow
from typing import List, Optional
@dataclass
class ElectronBS(Maker):
    """
    Maker for generating electron band structure data using Quantum ESPRESSO.
    """
    name: str = "ElectronBS"
    qe_run_cmd: str = "mpirun -np 1 pw.x"
    qe_bands_cmd: str = "bands.x"
    num_qe_workers: int | None = None
    num_bands_workers: int | None = None
    fname_scf_template: str | None = None
    fname_nscf_template: str | None = None
    fname_bands_template: str | None = None
    kspace_resolution: float | None = None 
    koffset: list[bool] = field(default_factory=lambda: [False, False, False]) 
    scf_outdir: List[dict] | None = None
    nscf_outdir: List[dict] | None = None
    run_scf: bool = True
    run_nscf: bool = True

    def __post_init__(self):
        if self.run_nscf and self.scf_outdir is None and not self.run_scf:
            raise ValueError("To run the nscf calculation either a scf calculation of the outdirs is needed.")
        if self.nscf_outdir is None and not self.run_nscf and not self.nscf_outdir:
            raise ValueError("To run the bands calculation either a nscf calculation of the outdirs is needed.")

    def make(self,structure_file: Optional[str] = None) -> Flow:
        """
        Create the flow to generate electron band structure data.
        """

        jobs = []

        if self.run_scf:
            # Create the job for static calculation
            scf_job = QEscf(dict(
                name="Static Calculation",
                qe_run_cmd=self.qe_run_cmd,
                num_qe_workers=self.num_qe_workers,
                fname_pwi_template=self.fname_scf_template,
                fname_structures=structure_file,
                kspace_resolution=self.kspace_resolution,
                koffset = self.koffset
            ))

            jobs.append(scf_job)
        
        if self.run_nscf:
            # Create the job for nscf calculation
            nscf_job = QEnscf(
                name="NSCF Calculation",
                nscf_run_command=self.qe_run_cmd,
                num_workers=self.num_qe_workers,
                fname_nscf_template=self.fname_nscf_template,
                scf_outdir=scf_job.output if self.run_scf else self.scf_outdir
            )

            jobs.append(nscf_job)

        # Create the job for band structure calculation
        band_job = QEband(
            name="Band Structure Calculation",
            bands_run_command=self.qe_bands_cmd,
            num_qe_workers=self.num_bands_workers,
            fname_pwi_template=self.fname_bands_template,
            scf_outdir=scf_job.output if self.run_scf else self.nscf_outdir,
            meta = {"dependency": nscf_job.output if self.run_nscf else None}
        )

        jobs.append(band_job)

        flow = Flow(jobs=jobs, output=band_job.output, name=self.name)


        return flow




