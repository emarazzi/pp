from pymatgen.core import Structure
import numpy as np
from typing import List, Tuple
from pymatgen.io.vasp.inputs import Kpoints
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor as aaa

pseudos: dict = dict(
	H = dict(mass = 1.0079, ecutwfc = 72.0),
	He = dict(mass = 4.0026, ecutwfc = 90.0),
	Li = dict(mass = 6.941, ecutwfc = 74.0),
	Be = dict(mass = 9.0122, ecutwfc = 88.0),
	B = dict(mass = 10.811, ecutwfc = 76.0),
	C = dict(mass = 12.0107, ecutwfc = 82.0),
	N = dict(mass = 14.0067, ecutwfc = 84.0),
	O = dict(mass = 15.9994, ecutwfc = 84.0),
	F = dict(mass = 18.9984, ecutwfc = 84.0),
	Ne = dict(mass = 20.1797, ecutwfc = 68.0),
	Na = dict(mass = 22.9897, ecutwfc = 88.0),
	Mg = dict(mass = 24.305, ecutwfc = 84.0),
	Al = dict(mass = 26.9815, ecutwfc = 40.0),
	Si = dict(mass = 28.0855, ecutwfc = 36.0),
	P = dict(mass = 30.9738, ecutwfc = 44.0),
	S = dict(mass = 32.065, ecutwfc = 52.0),
	Cl = dict(mass = 35.453, ecutwfc = 58.0),
	K = dict(mass = 39.0983, ecutwfc = 74.0),
	Ar = dict(mass = 39.948, ecutwfc = 66.0),
	Ca = dict(mass = 40.078, ecutwfc = 68.0),
	Sc = dict(mass = 44.9559, ecutwfc = 78.0),
	Ti = dict(mass = 47.867, ecutwfc = 84.0),
	V = dict(mass = 50.9415, ecutwfc = 84.0),
	Cr = dict(mass = 51.9961, ecutwfc = 94.0),
	Mn = dict(mass = 54.938, ecutwfc = 96.0),
	Fe = dict(mass = 55.845, ecutwfc = 90.0),
	Ni = dict(mass = 58.6934, ecutwfc = 98.0),
	Co = dict(mass = 58.9332, ecutwfc = 96.0),
	Cu = dict(mass = 63.546, ecutwfc = 92.0),
	Zn = dict(mass = 65.39, ecutwfc = 84.0),
	Ga = dict(mass = 69.723, ecutwfc = 80.0),
	Ge = dict(mass = 72.64, ecutwfc = 78.0),
	As = dict(mass = 74.9216, ecutwfc = 84.0),
	Se = dict(mass = 78.96, ecutwfc = 86.0),
	Br = dict(mass = 79.904, ecutwfc = 46.0),
	Kr = dict(mass = 83.8, ecutwfc = 52.0),
	Rb = dict(mass = 85.4678, ecutwfc = 46.0),
	Sr = dict(mass = 87.62, ecutwfc = 68.0),
	Y = dict(mass = 88.9059, ecutwfc = 72.0),
	Zr = dict(mass = 91.224, ecutwfc = 66.0),
	Nb = dict(mass = 92.9064, ecutwfc = 82.0),
	Mo = dict(mass = 95.94, ecutwfc = 80.0),
	Tc = dict(mass = 98.0, ecutwfc = 84.0),
	Ru = dict(mass = 101.07, ecutwfc = 84.0),
	Rh = dict(mass = 102.9055, ecutwfc = 88.0),
	Pd = dict(mass = 106.42, ecutwfc = 82.0),
	Ag = dict(mass = 107.8682, ecutwfc = 82.0),
	Cd = dict(mass = 112.411, ecutwfc = 102.0),
	In = dict(mass = 114.818, ecutwfc = 70.0),
	Sn = dict(mass = 118.71, ecutwfc = 72.0),
	Sb = dict(mass = 121.76, ecutwfc = 80.0),
	I = dict(mass = 126.9045, ecutwfc = 70.0),
	Te = dict(mass = 127.6, ecutwfc = 80.0),
	Xe = dict(mass = 131.293, ecutwfc = 68.0),
	Cs = dict(mass = 132.9055, ecutwfc = 50.0),
	Ba = dict(mass = 137.327, ecutwfc = 44.0),
	La = dict(mass = 138.9055, ecutwfc = 110.0),
	Ce = dict(mass = 140.116, ecutwfc = 30.0),
	Pr = dict(mass = 140.9077, ecutwfc = 30.0),
	Nd = dict(mass = 144.24, ecutwfc = 30.0),
	Pm = dict(mass = 145.0, ecutwfc = 30.0),
	Sm = dict(mass = 150.36, ecutwfc = 30.0),
	Eu = dict(mass = 151.964, ecutwfc = 30.0),
	Gd = dict(mass = 157.25, ecutwfc = 30.0),
	Tb = dict(mass = 158.9253, ecutwfc = 30.0),
	Dy = dict(mass = 162.5, ecutwfc = 30.0),
	Ho = dict(mass = 164.9303, ecutwfc = 30.0),
	Er = dict(mass = 167.259, ecutwfc = 30.0),
	Tm = dict(mass = 168.9342, ecutwfc = 30.0),
	Yb = dict(mass = 173.04, ecutwfc = 30.0),
	Lu = dict(mass = 174.967, ecutwfc = 100.0),
	Hf = dict(mass = 178.49, ecutwfc = 58.0),
	Ta = dict(mass = 180.9479, ecutwfc = 58.0),
	W = dict(mass = 183.84, ecutwfc = 74.0),
	Re = dict(mass = 186.207, ecutwfc = 72.0),
	Os = dict(mass = 190.23, ecutwfc = 74.0),
	Ir = dict(mass = 192.217, ecutwfc = 68.0),
	Pt = dict(mass = 195.078, ecutwfc = 84.0),
	Au = dict(mass = 196.9665, ecutwfc = 76.0),
	Hg = dict(mass = 200.59, ecutwfc = 66.0),
	Tl = dict(mass = 204.3833, ecutwfc = 62.0),
	Pb = dict(mass = 207.2, ecutwfc = 56.0),
	Bi = dict(mass = 208.9804, ecutwfc = 66.0),
	Po = dict(mass = 209.0, ecutwfc = 64.0),
	At = dict(mass = 210.0, ecutwfc = 30.0),
	Rn = dict(mass = 222.0, ecutwfc = 72.0),
	Fr = dict(mass = 223.0, ecutwfc = 30.0),
	Ra = dict(mass = 226.0, ecutwfc = 30.0),
	Ac = dict(mass = 227.0, ecutwfc = 30.0),
	Pa = dict(mass = 231.0359, ecutwfc = 30.0),
	Th = dict(mass = 232.0381, ecutwfc = 30.0),
	Np = dict(mass = 237.0, ecutwfc = 30.0),
	U = dict(mass = 238.0289, ecutwfc = 30.0),
	Am = dict(mass = 243.0, ecutwfc = 30.0),
	Pu = dict(mass = 244.0, ecutwfc = 30.0),
	Cm = dict(mass = 247.0, ecutwfc = 30.0),
	Bk = dict(mass = 247.0, ecutwfc = 30.0),
	Cf = dict(mass = 251.0, ecutwfc = 30.0),
	Es = dict(mass = 252.0, ecutwfc = 30.0),
	Fm = dict(mass = 257.0, ecutwfc = 30.0),
	Md = dict(mass = 258.0, ecutwfc = 30.0),
	No = dict(mass = 259.0, ecutwfc = 30.0),
	Rf = dict(mass = 261.0, ecutwfc = 30.0),
	Lr = dict(mass = 262.0, ecutwfc = 30.0),
	Db = dict(mass = 262.0, ecutwfc = 30.0),
	Bh = dict(mass = 264.0, ecutwfc = 30.0),
	Sg = dict(mass = 266.0, ecutwfc = 30.0),
	Mt = dict(mass = 268.0, ecutwfc = 30.0),
	Rg = dict(mass = 272.0, ecutwfc = 30.0),
	Hs = dict(mass = 277.0, ecutwfc = 30.0),

)




def write_qe_input(structure: Structure | Atoms, filename: str = "scf.in", prefix: str = 'qe', outdir: str = './',
                   degauss: float = 0.04, ecutwfc: float | None = None, 
                   conv_thr: float | None = None, electron_maxstep: int = 100, mixing_beta: float = 0.4,
                   kgrid: List | Tuple | None = None, kppa: int | None = 1000, shiftk: List | Tuple = [0,0,0], 
                   is_2D: bool = False, is_1D: bool = False):
    """
    Write a QE scf input file from a pymatgen Structure object.
    """
    if isinstance(structure, Atoms):
        structure = aaa.get_structure(structure)
    coords = structure.cart_coords
    species = structure.species
    lattice = structure.lattice.matrix
    symbols = structure.symbol_set
    if kgrid is None:
        kpts = Kpoints.automatic_density(structure=structure,kppa=kppa).as_dict()
        kgrid = kpts['kpoints'][0]
        shiftk = kpts['usershift']
    if ecutwfc is None:
        ecutwfc = np.max([pseudos[s]['ecutwfc'] for s in symbols])

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
  calculation 		= 	'scf'
  etot_conv_thr 	=   1.0000000000d-05
  forc_conv_thr 	=   5.0000000000d-05
  outdir 			= 	'{outdir}'
  prefix 			= 	'{prefix}'
  tprnfor 			= 	.true.
  tstress			= 	.true.
/
&SYSTEM
  ibrav            	=	0
  degauss          	=   {degauss}
  ecutrho          	=   {ecutwfc*4}
  ecutwfc          	=   {ecutwfc}
  nat              	= 	{int(len(structure))}
  nosym            	= 	.false.
  ntyp             	= 	{len(symbols)}
  occupations      	= 	'smearing'
  smearing         	= 	'cold'
  assume_isolated	= 	'{assume_isolated}'
/
&ELECTRONS
  conv_thr 			=   {conv_thr}
  electron_maxstep	= 	{electron_maxstep}
  mixing_beta 		=  	{mixing_beta}
/
ATOMIC_SPECIES
""")
        for symbol in symbols:
            file.write(f"{symbol} {pseudos[symbol]['mass']} {symbol+'.upf'}\n")
        file.write("ATOMIC_POSITIONS angstrom\n")
        for s,c in zip(species,coords):
            file.write(f"{s.symbol}     {c[0]:.12f} {c[1]:.12f} {c[2]:.12f}\n")
        file.write(f"K_POINTS automatic\n")
        file.write(f"{kgrid[0]}  {kgrid[1]}  {kgrid[2]}  {shiftk[0]}  {shiftk[1]}  {shiftk[2]}\n")
        file.write(f"CELL_PARAMETERS angstrom\n")
        for l in lattice:
            file.write(f"{l[0]:.12f} {l[1]:.12f} {l[2]:.12f} \n")

def write_pw2bgw(kgrid:List|Tuple, filename:str="pw2bgw.in", prefix:str='qe', outdir:str='./', vscg_file:str='VSC',shiftk:List=[0,0,0]):
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
