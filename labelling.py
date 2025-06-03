"""Jobs to create training data for ML potentials."""

import os
import logging
import subprocess
from glob import glob
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
import numpy as np

from ase import Atoms
from ase.io import read, write
from jobflow import job, Flow, Maker, Response


from pymatgen.core import Structure
from pp.dft_inputs import write_qe_input



logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class QEstaticLabelling(Maker):
    """
    Maker to set up and run Quantum Espresso static calculations for input structures, including bulk, isolated atoms, and dimers.
    Parameters
    ----------
    name: str
        Name of the flow.
    qe_run_cmd: str
        String with the command to run QE (including its executable path/or application name).
    fname_pwi_template: str
        Path to file containing the template computational parameters.
    fname_structures: str
        Path to ASE-readible file containing the structures to be computed.
    num_qe_workers: int | None
        Number of workers to use for the calculations. If None, defaults to the number of structures.
    """

    name: str = "do_qe_labelling"
    qe_run_cmd: str | None = None #String with the command to run QE (including its executable path/or application name)
    fname_pwi_template: str | None = None #Path to file containing the template computational parameters
    fname_structures: str | None = None #Path to ASE-readible file containing the structures to be computed
    num_qe_workers: int | None = None #Number of workers to use for the calculations. 

    def make(self, **qe_kwargs):
        #Define jobs
        joblist = []

        # Load structures
        if self.fname_structures is None: raise ValueError("No structure paths provided. Please provide a list of paths to structures.")
        #structures = read(self.fname_structures, index=":")
        structures = [Structure.from_file(structure) for structure in self.fname_structures.split(':')]
        if len(structures) == 0: raise ValueError("No structures found in the provided file. Please provide a valid file with structures.")

        # Check pwi template
        #pwi_template_lines = self.check_pwi_template(self.fname_pwi_template)

        # Write pwi input files for each structure
        work_dir = os.getcwd()
        path_to_qe_workdir = os.path.join(work_dir, "scf_files")
        os.makedirs(path_to_qe_workdir, exist_ok=True)

        for i, structure in enumerate(structures):
            fname_new_pwi = os.path.join(path_to_qe_workdir, f"structure_{i}.pwi")
            #self.write_pwi(
            #    fname_pwi_output=fname_new_pwi,
            #    structure=structure, 
            #    pwi_template=pwi_template_lines, 
            #    )
            write_qe_input(structure=structure, filename=fname_new_pwi, **qe_kwargs)

        # Set number of QE workers
        if self.num_qe_workers is None: # 1 worker per structure (all DFT jobs in parallel)
            num_qe_workers = len(glob(os.path.join(path_to_qe_workdir, "*.pwi")))
        else: 
            num_qe_workers = self.num_qe_workers

        # Launch QE workers            
        success_of_workers = []
        for id_qe_worker in range(num_qe_workers):
           qe_worker = self.run_qe_worker(
               id=id_qe_worker,
               command=self.qe_run_cmd,
               work_dir=path_to_qe_workdir
               )
           
           qe_worker.name = f"run_qe_worker_{id_qe_worker}"
           joblist.append(qe_worker)
           success_of_workers.append(qe_worker.output)

        # Output is a list of success status, one for each worker
        # The success status is a dictionary with the pwo file name as key and the calculation success status as value (True/False)
        return Response(replace=Flow(joblist), output=success_of_workers)

    def check_pwi_template(self, fname_template):
        """
        Check the pwi template file for the required parameters.
        """
        # Read template file
        tmp_pwi_lines = []
        with open(fname_template, 'r') as f:
            tmp_pwi_lines = f.readlines()

        # Modify lines with structure information: 
        # Assume ntyp, atom_types and pseudoptentials are already defined in the template and consistent with the structures
        # Assume ibrav=0 and Kspacing is already defined in the template
        idx_nat_line, idx_kpoints_line, idx_pos_line, idx_cell_line = 0, 0, 0, 0
        for i, line in enumerate(tmp_pwi_lines):
            if 'nat' in line: idx_nat_line = i
            
            elif 'K_POINTS' in line: idx_kpoints_line = i

            elif 'ATOMIC_POSITIONS' in line: idx_pos_line = i

            elif 'CELL_PARAMETERS' in line: idx_cell_line = i
        
        # Set nat line
        if idx_nat_line == 0: # nat not defined, assume nat = 0
            raise ValueError("Number of atoms line not defined in the template file. Please define \'nat =\' in the template file.")
        else:
            tmp_pwi_lines[idx_nat_line] = f'nat = \n'
        
        # Set K_points lines
        # TODO: Set K_points lines based on the structure and K-spacing
        if idx_kpoints_line == 0: # K_POINTS not defined, assume Gamma point
            kpoints_lines = ["\nK_POINTS Gamma\n"]
        elif idx_kpoints_line > 0:
            kpoints_lines = tmp_pwi_lines[idx_kpoints_line:idx_kpoints_line+1]
            del tmp_pwi_lines[idx_kpoints_line:]

        # Cancel lines with ATOMIC_POSITIONS and CELL_PARAMETERS
        if idx_pos_line == 0 and idx_cell_line > 0:
            idx_to_delete = idx_pos_line
            del(tmp_pwi_lines[idx_to_delete:])
        
        elif idx_pos_line > 0 and idx_cell_line == 0:
            idx_to_delete = idx_pos_line
            del(tmp_pwi_lines[idx_to_delete:])
        
        elif idx_pos_line > 0 and idx_cell_line > 0:
            idx_to_delete = min([idx_pos_line, idx_cell_line])
            del(tmp_pwi_lines[idx_to_delete:])
        
        # Build final template lines
        tmp_pwi_lines = tmp_pwi_lines + kpoints_lines

        return tmp_pwi_lines

    def write_pwi(
            self, 
            fname_pwi_output: str,
            structure: Atoms, 
            pwi_template: list[str],
            ):
        """
        Write the pwi input file for the given structure.
        """
        # Write nat line
        idx_nat_line, nat = 0, len(structure)
        for idx, line in enumerate(pwi_template):
            if 'nat =' in line: idx_nat_line = idx
        pwi_template[idx_nat_line] = f'nat = {nat}\n'

        #Write cell lines
        cell_lines = ["\nCELL_PARAMETERS (angstrom)\n"]
        cell_lines += [f"{structure.cell[i, 0]:.10f} {structure.cell[i, 1]:.10f} {structure.cell[i, 2]:.10f}\n" for i in range(3)]
        
        #Write positions lines
        pos_lines = ["\nATOMIC_POSITIONS (angstrom)\n"]
        for i, atom in enumerate(structure):
            pos_lines.append(f"{atom.symbol} {structure.positions[i, 0]:.10f} {structure.positions[i, 1]:.10f} {structure.positions[i, 2]:.10f}\n")

        # Write the modified lines to the new pwi file
        with open(fname_pwi_output, 'w') as f:
            for line in pwi_template:
                f.write(line)
            for line in cell_lines:
                f.write(line)
            for line in pos_lines:
                f.write(line)

    @job
    def run_qe_worker(
            self, 
            id,
            command,
            work_dir,
            ):
        """
        Run the QE command in a subprocess.
        """
        #Get pwi files
        pwi_files = glob(os.path.join(work_dir, "*.pwi"))

        #Check pwo does not exist
        success_pwo = {}
        for pwi in pwi_files:
            #Try locking the pwi file
            lock_pwi, pwo_fname = self.lock_input(pwi_fname=pwi, worker_id=id)

            if lock_pwi == "": continue #Skip to next pwi if lock failed

            #Launch QE calculation
            success = self.run_qe(command=command, fname_pwi=lock_pwi, fname_pwo=pwo_fname)

            #Set success status
            success_pwo[pwo_fname] = success

        return success_pwo

    def run_qe(self, command, fname_pwi, fname_pwo):
        """
        Run the QE command in a subprocess. Execute one QuantumEspresso calculation on the current input file.
        """
        #Assemble QE command
        run_cmd = f"{command} < {fname_pwi} >> {fname_pwo}"

        success = False
        try:        
            # Launch QE and wait till ending
            subprocess.run(run_cmd, shell=True, check=True, executable="/bin/bash")
            
            success = True
        
        except subprocess.CalledProcessError as e:
            
            success = False

        return success
    
    def lock_input(self, pwi_fname, worker_id):
        
        pwi_lock_fname = ""
        #Check if pwo exists
        pwo_fname = pwi_fname.replace('.pwi', '.pwo')
        if os.path.exists(pwo_fname): return pwi_lock_fname, pwo_fname #If exists, skip to next pwi

        # Try to lock the pwi file by renaming it
        pwi_lock_fname = f'{pwi_fname}.lock_{worker_id}'
        try:
            os.rename(f'{pwi_fname}', f'{pwi_lock_fname}')
        except Exception as e:
            pwi_lock_fname = ""  
        
        return pwi_lock_fname, pwo_fname
