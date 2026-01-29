import numpy as np
from HPRO.structure import load_structure
from HPRO.lcaodata import LCAOData
from HPRO.orbutils import OrbPair
from HPRO.twocenter import calc_overlap
from HPRO.deephio import save_mat_deeph, load_deeph_HS
import h5py
from scipy.sparse.linalg import lsmr
from tqdm import tqdm
import sys
from scipy.sparse import coo_matrix
from HPRO.matlcao import MatLCAO


def compute_kinetic_energy_only(
        structure_path: str, 
        structure_interface: str, 
        lcaodata_root: str, 
        lcao_interface: str, 
        ecutwfn: int | float, 
        kinetic_energy_filename: str = 'kinetic_energy.h5',
        savedir: str | None = './',
        energy_unit: bool = True
        ) -> MatLCAO:
    # 1. Load the material structure 
    structure = load_structure(structure_path, structure_interface)
    
    # 2. Initialise the LCAO basis data
    basis = LCAOData(structure, basis_path_root=lcaodata_root, aocode=lcao_interface)
    basis.check_rstart() 
    
    # 3. Prepare Fourier transform of orbitals for the given cutoff
    ecut = ecutwfn
    basis.calc_phiQ(ecut * 1.1) 
    
    # 4. Generate orbital pairs specifically for kinetic energy (index=2) 
    # This loop follows the logic for 'orbpairs3' in the sources 
    orbpairs_kin = {}
    for ispc in range(structure.nspc):
        for jspc in range(structure.nspc):
            spc1 = structure.atomic_species[ispc]
            spc2 = structure.atomic_species[jspc]
            
            pairs_list = []
            for jorb in range(basis.norb_spc[spc2]):
                r2 = basis.phirgrids_spc[spc2][jorb].rcut
                for iorb in range(basis.norb_spc[spc1]):
                    r1 = basis.phirgrids_spc[spc1][iorb].rcut
                    
                    # index=2 triggers kinetic energy matrix element calculation
                    thispair = OrbPair(basis.phiQlist_spc[spc1][iorb],
                                     basis.phiQlist_spc[spc2][jorb], 
                                     r1 + r2, index=2)
                    pairs_list.append(thispair)
            
            orbpairs_kin[(spc1, spc2)] = pairs_list

    # 5. Calculate the Kinetic Energy matrix (Hkin) 
    # This uses the two-center integration scheme 
    h_kin = calc_overlap(basis, orbpairs_kin, Ecut=ecut)
    
    # 6. Save the resulting matrix
    if savedir:
        save_mat_deeph(savedir, h_kin, filename=kinetic_energy_filename, energy_unit=energy_unit)
    
    return h_kin

def compute_potential_ij(
        matrices_path: str,
        hamiltonians_filename: str = 'hamiltonians.h5',
        kinetic_energy_filename: str = 'kinetic_energy.h5',
        potential_filename: str = 'potential_energy.h5',
        energy_unit: bool = True
        ) -> HPRO.matlcao.MatLCAO:
    

    mat_h = load_deeph_HS(matrices_path, hamiltonians_filename, energy_unit=energy_unit)
    mat_t = load_deeph_HS(matrices_path, kinetic_energy_filename, energy_unit=energy_unit)

    mat_v = mat_h - mat_t

    save_mat_deeph(matrices_path, mat_v, filename=potential_filename, energy_unit=energy_unit)

    return mat_v

def compute_real_space(
        foldername: str = './',
        interface: str = 'deeph',
        ij_filename: str = 'potential_energy.h5',
        ao_basis_folder: str = './',
        output_filename: str = 'Vr.xyzv',
        grid_shape: list | tuple = (5, 5, 5),
        shift_cart: np.ndarray = np.zeros(3)
        ) -> None:


    # --- 1. LOAD STRUCTURE DATA ---
    structure = load_structure(foldername,interface)
    lcaodata = LCAOData(structure=structure,basis_path_root=ao_basis_folder)


    # --- 2. CONFIGURATION & GRID ---
    nx, ny, nz = grid_shape
    rprim = structure.rprim

    gx, gy, gz = np.meshgrid(np.arange(nx)/nx, np.arange(ny)/ny, np.arange(nz)/nz, indexing='ij')
    grid_frac = np.stack((gx.flatten(), gy.flatten(), gz.flatten()), axis=1)

    inv_rprim = np.linalg.inv(rprim)
    shift_frac = shift_cart @ inv_rprim
    
    grid_frac_shifted = (grid_frac + shift_frac) % 1.0
    rp_cart = grid_frac_shifted @ rprim
    
    n_grid_points = rp_cart.shape[0]

    # --- 3. LOAD DATA ---

    ij = []
    with h5py.File(ij_filename, 'r') as hf:
        ij_keys = list(hf.keys())
        for key in ij_keys:
            ij.extend(hf[key][:].flatten())
    
    b = np.array(ij)
    
    n_observations = b.shape[0]


    # --- 4. CONSTRUCT THE SYSTEM MATRIX A ---
    # A will be (n_observations rows x n_grid_points columns)
    print(f"Building Matrix A: {n_observations} x {n_grid_points}...")
    
    data_list = []
    row_list = []
    col_list = []
    current_row = 0
    
    SPARSITY_THRESHOLD = 1e-8



    for k in tqdm(ij_keys, file=sys.stdout, mininterval=60, desc="Building Matrix A"):
        key = k.strip('[]')
        k_clean = [int(k) for k in key.split(',')]
        translation_j = k_clean[:3]
        i_atm = k_clean[3]-1
        j_atm = k_clean[4]-1
        spc1 = structure.atomic_numbers[i_atm]
        spc2 = structure.atomic_numbers[j_atm]
        phi_i_func = lcaodata.phirgrids_spc[spc1] 
        phi_j_func = lcaodata.phirgrids_spc[spc2]
        
        # 5. Evaluate orbitals at grid points r_p
        # We must subtract the atomic position (and translation R) from the grid point
        pos_i = structure.atomic_positions_cart[i_atm] 
        pos_j = structure.atomic_positions_cart[j_atm]
        
        if not np.allclose(translation_j,np.zeros(3)):
            # If the second orbital is in a different unit cell (R)
            pos_j = pos_j + translation_j @ structure.rprim
        
        # generate3D returns values at each point p for each m in (2l+1)
        val_i = np.array([phi.generate3D(rp_cart - pos_i)[:,j] for phi in phi_i_func
                          for j in range(len(phi.generate3D(np.array([0,0,0]))))])
    
        val_j = np.array([phi.generate3D(rp_cart - pos_j)[:,j] for phi in phi_j_func
                          for j in range(len(phi.generate3D(np.array([0,0,0]))))])
            
        # 6. Compute the product Mp,ij
    
        m_p_ij = val_i[:, np.newaxis, :] * val_j[np.newaxis, :, :]
        m_p_ij = m_p_ij.reshape(-1, n_grid_points) * (1.0 / n_grid_points)  
        
        for row_in_block in range(m_p_ij.shape[0]):
            row_values = m_p_ij[row_in_block, :]
            
            # Find indices where values are significant
            significant_indices = np.where(np.abs(row_values) > SPARSITY_THRESHOLD)[0]
            
            if len(significant_indices) > 0:
                data_list.extend(row_values[significant_indices])
                col_list.extend(significant_indices)
                # All these values belong to the same global row in A
                row_list.extend([current_row] * len(significant_indices))
            
            current_row += 1
    print("Converting to CSR format...")
    A_sparse = coo_matrix((data_list, (row_list, col_list)), 
                          shape=(n_observations, n_grid_points)).tocsr()
    
    # Free up the lists from memory
    del data_list, row_list, col_list
    # --- 8. SOLVE THE SYSTEM ---


    print("Solving overdetermined least-squares system...")
    
    solution_data = lsmr(A_sparse, b, damp=0.1, show=True)
    V_solution = solution_data[0]
    # --- 9. SAVE RESULTS TO FILE ---
    V_col = V_solution[:, np.newaxis]
    
    data_to_save = np.hstack([rp_cart, V_col])
    
    np.savetxt(output_filename, data_to_save,
               header='x y z V', comments='', fmt='%.8e')


def extract_potential_in_real_space(
        structure_path: str = './',
        structure_interface: str = 'deeph',
        lcaodata_root: str = './',
        lcao_interface: str = 'siesta',
        ecutwfn: int | float = 100,
        savedir: str | None = './',
        hamiltonians_filename: str = 'hamiltonians.h5',
        potential_filename: str = 'potential_energy.h5',
        matrices_path: str = './',
        energy_unit: bool = True,
        grid_shape: list | tuple = (5,5,5),
        shift_cart: np.ndarray = np.zeros(3),
        ao_basis_folder: str = './',
        output_filename: str = 'Vr.xyzv'                         
):

    mat_k = compute_kinetic_energy_only(
        structure_path=structure_path, 
        structure_interface=structure_interface, 
        lcaodata_root=lcaodata_root, 
        lcao_interface=lcao_interface, 
        ecutwfn=ecutwfn, 
        savedir=savedir
    )

    mat_h = load_deeph_HS(matrices_path, hamiltonians_filename, energy_unit=energy_unit)

    mat_v = mat_h - mat_k

    save_mat_deeph(matrices_path, mat_v, filename=potential_filename, energy_unit=energy_unit)

    compute_real_space(foldername=structure_path,
        interface = structure_interface,
        ij_filename = potential_filename,
        ao_basis_folder = ao_basis_folder,
        output_filename = output_filename,
        grid_shape = grid_shape,
        shift_cart = shift_cart
        )



