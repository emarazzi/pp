from pymatgen.core import Structure
import numpy as np
from typing import List, Tuple
from pymatgen.io.vasp.inputs import Kpoints
        
def write_qe_input(structure: Structure, pseudos = dict, filename: str = "scf.in", prefix:str='qe', outdir:str='./',
        degauss:float=0.04, ecutwfc:float=30.0, conv_thr:float=None, electron_maxstep:int=100, mixing_beta:float=0.4,
        kgrid:List|Tuple=None, kppa:int|None=1000, shiftk:List|Tuple=[0,0,0], is_2D:bool=False, is_1D:bool=False):
    """
    Write a QE scf input file from a pymatgen Structure object.
    """
    coords = structure.cart_coords
    species = structure.species
    lattice = structure.lattice.matrix
    symbols = structure.symbol_set
    if kgrid is None:
        kpts = Kpoints.automatic_density(structure=structure,kppa=1000).as_dict()
        kgrid = kpts['kpoints'][0]
        shiftk = kpts['usershift']

    if conv_thr is None:
        conv_thr = 2.0e-10*len(structure)
    if is_2D:
        kgrid = [kgrid[0], kgrid[1], 1]
        assume_isolated = '2D'
        lattice = [[max(11,abs(max(coords[:,2])-min(coords[:,2]))+5.5) if i==j and i ==2 else lattice[i][j] for j in range(3)] for i in range(3)]
    else:
        assume_isolated = 'none'
    if is_1D:
        kgrid = [1, 1, kgrid[2]]

    with open(filename,"w") as file:
        file.write(f"""&CONTROL
  calculation = 'scf'
  etot_conv_thr =   1.0000000000d-05
  forc_conv_thr =   5.0000000000d-05
  outdir = '{outdir}'
  prefix = '{prefix}'
  pseudo_dir = {pseudos['pseudo_dir']}
  tprnfor = .true.
  tstress = .true.
/
&SYSTEM
  ibrav =  0
  degauss =   {degauss}
  ecutrho =   {ecutwfc*4}
  ecutwfc =   {ecutwfc}
  nat = {int(len(structure))}
  nosym = .false.
  ntyp = {len(symbols)}
  occupations = 'smearing'
  smearing = 'cold'
  assume_isolated = '{assume_isolated}'
/
&ELECTRONS
  conv_thr =   {conv_thr}
  electron_maxstep = {electron_maxstep}
  mixing_beta =   {mixing_beta}
/
ATOMIC_SPECIES
""")
        for symbol in symbols:
            file.write(f"{symbol} {pseudos[symbol]['mass']} {pseudos[symbol]['filename']}\n")
        file.write("ATOMIC_POSITIONS angstrom\n")
        for s,c in zip(species,coords):
            file.write(f"{s.symbol}     {c[0]:.12f} {c[1]:.12f} {c[2]:.12f}\n")
        file.write(f"K_POINTS automatic\n")
        file.write(f"{kgrid[0]}  {kgrid[1]}  {kgrid[2]}  {shiftk[0]}  {shiftk[1]}  {shiftk[2]}\n")
        file.write(f"CELL_PARAMETERS angstrom\n")
        for l in lattice:
            file.write(f"{l[0]:.12f} {l[1]:.12f} {l[2]:.12f} \n")

def write_pw2bgw(kgrid:List|Tuple, filename:str="pw2bgw.in", prefix:str='qe', outdir:str='./', vscg_file:str='VSC'):
    with open(filename,'w') as file:
        file.write(f"""&input_pw2bgw
   prefix = '{prefix}'
   outdir = '{outdir}'
   real_or_complex = 2
   wfng_flag = .false.
   wfng_kgrid = .true.
   wfng_nk1 = {kgrid[0]}
   wfng_nk2 = {kgrid[1]}
   wfng_nk3 = {kgrid[2]}
   wfng_dk1 = {shiftk[0]}
   wfng_dk2 = {shiftk[1]}
   wfng_dk3 = {shiftk[2]}

   vscg_flag = .true.
   vscg_file = '{vscg_file}'
/""")
