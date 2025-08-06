from pp.jobs.jobs import QEscf, QEband, QEnscf

from dataclasses import dataclass, field
from jobflow import Maker, Flow
from typing import List, Optional

@dataclass
class ScfSettings:
    qe_run_cmd: str = "mpirun -np 1 pw.x"
    num_qe_workers: Optional[int] = None
    fname_scf_template: Optional[str] = None
    kspace_resolution: Optional[float] = None
    koffset: List[bool] = field(default_factory=lambda: [False, False, False])

@dataclass
class NscfSettings:
    qe_run_cmd: str = "mpirun -np 1 pw.x"
    num_qe_workers: Optional[int] = None
    fname_nscf_template: Optional[str] = None
    scf_outdir: Optional[List[dict]] = None
    hs_path: Optional[List] = None
    ndivsm: Optional[int] = None

@dataclass
class BandsSettings:
    qe_bands_cmd: str = "bands.x"
    num_bands_workers: Optional[int] = None
    fname_bands_template: Optional[str] = None
    scf_outdir: Optional[List[dict]] = None

@dataclass
class ElectronBS(Maker):
    """
    Maker for generating electron band structure data using Quantum ESPRESSO.
    """
    name: str = "ElectronBS"
    scf_settings: ScfSettings = field(default_factory=ScfSettings)
    nscf_settings: NscfSettings = field(default_factory=NscfSettings)
    bands_settings: BandsSettings = field(default_factory=BandsSettings)

    def __post_init__(self):
        if self.nscf_settings and not (self.nscf_settings.scf_outdir or self.scf_settings):
            raise ValueError("To run NSCF, provide SCF output or enable SCF calculation.")
        if not self.nscf_settings and self.bands_settings.scf_outdir is None:
            raise ValueError("To run bands calculation, provide NSCF or SCF output.")

    def make(self, structure_file: Optional[str] = None) -> Flow:
        """
        Create the flow to generate electron band structure data.

        Args:
            structure_file (Optional[str]): Path to the structure file

        Returns:
            Flow: A JobFlow Flow for band structure computation
        """

        jobs = []

        if self.scf_settings:
            # Create the job for static calculation
            scf_job = QEscf(dict(
                name="Static Calculation",
                qe_run_cmd=self.scf_settings.qe_run_cmd,
                num_qe_workers=self.scf_settings.num_qe_workers,
                fname_pwi_template=self.scf_settings.fname_scf_template,
                fname_structures=structure_file,
                kspace_resolution=self.scf_settings.kspace_resolution,
                koffset = self.scf_settings.koffset
            ))

            jobs.append(scf_job)
        
        if self.nscf_settings:
            # Create the job for nscf calculation
            nscf_job = QEnscf(
                name="NSCF Calculation",
                nscf_run_command=self.nscf_settings.qe_run_cmd,
                num_workers=self.nscf_settings.num_qe_workers,
                fname_nscf_template=self.nscf_settings.fname_nscf_template,
                hs_path=self.nscf_settings.hs_path,
                ndivsm=self.nscf_settings.ndivsm,
                scf_outdir=scf_job.output if self.scf_settings else self.nscf_settings.scf_outdir
            )

            jobs.append(nscf_job)

        # Create the job for band structure calculation
        band_job = QEband(
            name="Band Structure Calculation",
            bands_run_command=self.bands_settings.qe_bands_cmd,
            num_qe_workers=self.bands_settings.num_bands_workers,
            fname_pwi_template=self.bands_settings.fname_bands_template,
            scf_outdir=scf_job.output if self.run_scf else self.bands_settings.scf_outdir,
            meta = {"dependency": nscf_job.output if self.nscf_settings else None}
        )

        jobs.append(band_job)

        flow = Flow(jobs=jobs, output=band_job.output, name=self.name)


        return flow




