"""
Utility functions for structure manipulation, k-path generation, and file I/O.

This module provides tools for generating training structures, converting between
formats, handling k-paths for band structure calculations, and reading/writing
various file formats used in electronic structure calculations.
"""

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from numpy import ndarray
import json
from pathlib import Path
from shutil import copy
import re
from jobflow import job


# Physical constants
BOHR_TO_ANGSTROM = 0.529177249


def _validate_file_exists(file_path: str | Path, param_name: str = "file") -> Path:
    """
    Validate that a file exists and return as Path object.

    Parameters
    ----------
    file_path : str | Path
        Path to the file to validate
    param_name : str
        Parameter name for error messages

    Returns
    -------
    Path
        Validated Path object

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the path is not a file
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{param_name} not found: {path}")

    if not path.is_file():
        raise ValueError(f"{param_name} is not a file: {path}")

    return path


def _validate_structure(structure: Structure, param_name: str = "structure") -> None:
    """
    Validate a pymatgen Structure object.

    Parameters
    ----------
    structure : Structure
        Structure to validate
    param_name : str
        Parameter name for error messages

    Raises
    ------
    ValueError
        If structure is invalid or empty
    """
    if not isinstance(structure, Structure):
        raise ValueError(f"{param_name} must be a pymatgen Structure object")

    if len(structure) == 0:
        raise ValueError(f"{param_name} is empty (no atoms)")


def _validate_positive(value: float | int, param_name: str, allow_zero: bool = False) -> None:
    """
    Validate that a numeric value is positive.

    Parameters
    ----------
    value : float | int
        Value to validate
    param_name : str
        Parameter name for error messages
    allow_zero : bool
        Whether to allow zero values

    Raises
    ------
    ValueError
        If value is not positive (or non-negative if allow_zero=True)
    """
    if allow_zero and value < 0:
        raise ValueError(f"{param_name} must be non-negative, got {value}")
    elif not allow_zero and value <= 0:
        raise ValueError(f"{param_name} must be positive, got {value}")


def generate_training_population(
    structure: Structure,
    structures_dir: str | Path,
    supercell_size: list[int] | None = None,
    distance: float = 0.1,
    min_distance: float | None = None,
    size: int = 200,
) -> list[str]:
    """
    Generate a set of perturbed structures starting from a reference Pymatgen Structure.

    This function takes an input Pymatgen Structure, expands it into a supercell,
    and generates multiple perturbed copies by randomly displacing atomic positions.
    Each perturbed structure is written to disk as a CIF file in the specified directory.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The input structure to use as a reference. This object will not be modified.
    structures_dir : str | Path
        Path to the directory where generated CIF files will be saved.
        The directory will be created if it does not exist.
    supercell_size : list[int], optional
        Supercell expansion factors along each lattice vector, e.g. [2, 2, 2].
        Default is [1, 1, 1].
    distance : float, optional
        Maximum displacement magnitude (in Å) applied to atomic positions
        during perturbation. Default is 0.1 Å.
    min_distance : float, optional
        Minimum displacement magnitude (in Å) applied to atomic positions
        during perturbation. Default is None, i.e. everything is perturbed by same amount.
    size : int, optional
        Number of perturbed structures to generate. Default is 200.

    Returns
    -------
    list[str]
        List of full file paths (CIF format) corresponding to the generated perturbed structures.

    Raises
    ------
    ValueError
        If input parameters are invalid

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> structure = Structure.from_file("POSCAR")
    >>> paths = generate_training_population(
    ...     structure, "training_structures", supercell_size=[2, 2, 2], size=100
    ... )
    >>> print(f"Generated {len(paths)} structures")
    """
    # Validate inputs
    _validate_structure(structure, "structure")
    _validate_positive(distance, "distance")
    if min_distance is not None:
        _validate_positive(min_distance, "min_distance")
        if min_distance > distance:
            raise ValueError(
                f"min_distance ({min_distance}) cannot be greater than distance ({distance})"
            )
    _validate_positive(size, "size")

    if supercell_size is None:
        supercell_size = [1, 1, 1]

    if len(supercell_size) != 3:
        raise ValueError(f"supercell_size must have 3 elements, got {len(supercell_size)}")

    if any(s <= 0 for s in supercell_size):
        raise ValueError(f"supercell_size elements must be positive, got {supercell_size}")

    # Ensure output directory exists
    structures_dir = Path(structures_dir)
    structures_dir.mkdir(parents=True, exist_ok=True)

    # Create a supercell reference structure
    base_structure = structure.copy()
    base_structure.make_supercell(supercell_size)

    structures_fname: list[str] = []

    # Save the unperturbed base structure
    fname = structures_dir / "0.cif"
    base_structure.to(fmt="cif", filename=str(fname))
    structures_fname.append(str(fname))

    # Generate perturbed structures
    for j in range(1, size):
        # Work with a fresh copy each time to avoid cumulative perturbations
        perturbed = base_structure.copy()
        perturbed.perturb(distance=distance, min_distance=min_distance)

        fname = structures_dir / f"{j}.cif"
        perturbed.to(filename=str(fname))
        structures_fname.append(str(fname))

    return structures_fname


def standard_primitive(file_in: str | Path, file_out: str | Path | None = None) -> Structure:
    """
    Convert a structure to its standard primitive cell.

    Reads a structure from file, determines its primitive standard form using
    spacegroup analysis, and writes it to a CIF file.

    Parameters
    ----------
    file_in : str | Path
        Path to input structure file (any format supported by pymatgen)
    file_out : str | Path | None, optional
        Path to output CIF file. If None, saves as "{input_stem}-prim.cif"
        in the same directory as the input file.

    Returns
    -------
    Structure
        The primitive standard structure

    Raises
    ------
    FileNotFoundError
        If input file does not exist

    Examples
    --------
    >>> prim_structure = standard_primitive("POSCAR", "primitive.cif")
    >>> prim_structure = standard_primitive("structure.cif")  # saves as structure-prim.cif
    """
    file_in = _validate_file_exists(file_in, "file_in")

    structure = Structure.from_file(str(file_in))
    structure_primitive = SpacegroupAnalyzer(structure).get_primitive_standard_structure()

    if file_out:
        file_out = Path(file_out)
        structure_primitive.to(str(file_out))
    else:
        # Use pathlib to handle file extensions properly
        output_path = file_in.parent / f"{file_in.stem}-prim.cif"
        structure_primitive.to(str(output_path))

    return structure_primitive


class KPath:
    """
    K-path generator for band structure calculations.

    This class handles the generation of k-point paths through the Brillouin zone
    for band structure calculations, with automatic spacing to ensure uniform
    point density along the path.

    Parameters
    ----------
    HSPoints : list
        List of fractional coordinates of high-symmetry points, shape (N, 3)
    avecs : ndarray
        3x3 array of real-space lattice vectors as rows (in Å)
    ndivsm : int, optional
        Number of divisions for the shortest segment (QE style). Default is 10.

    Attributes
    ----------
    HSPoints : ndarray
        Array of high-symmetry points in fractional coordinates
    avecs : ndarray
        Real-space lattice vectors (rows)
    bvecs : ndarray
        Reciprocal-space lattice vectors
    ndivsm : int
        Target number of divisions for shortest segment

    Examples
    --------
    >>> import numpy as np
    >>> # Define cubic lattice
    >>> avecs = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
    >>> # Gamma -> X -> M path
    >>> points = [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]]
    >>> kpath = KPath(points, avecs, ndivsm=20)
    >>> divisions = kpath.get_divisions()
    """

    def __init__(self, HSPoints: list, avecs: ndarray, ndivsm: int = 10):
        """Initialize KPath with high-symmetry points and lattice vectors."""
        self.HSPoints = np.array(HSPoints)
        self.avecs = np.array(avecs)
        self.ndivsm = ndivsm

        # Validate inputs
        if self.HSPoints.ndim != 2 or self.HSPoints.shape[1] != 3:
            raise ValueError(
                f"HSPoints must have shape (N, 3), got {self.HSPoints.shape}"
            )

        if self.avecs.shape != (3, 3):
            raise ValueError(
                f"avecs must have shape (3, 3), got {self.avecs.shape}"
            )

        if ndivsm <= 0:
            raise ValueError(f"ndivsm must be positive, got {ndivsm}")

        # Compute reciprocal lattice vectors: b_i = 2π (A^{-1})^T
        self.bvecs = 2 * np.pi * np.linalg.inv(self.avecs).T

    def get_divisions(self) -> ndarray:
        """
        Compute the number of divisions per segment.

        Returns the number of k-points for each segment so that the shortest
        segment has ndivsm divisions, maintaining uniform point density.

        Returns
        -------
        ndarray
            Array of integers representing divisions per segment

        Examples
        --------
        >>> kpath = KPath([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]], avecs, ndivsm=10)
        >>> divisions = kpath.get_divisions()
        """
        n_segments = len(self.HSPoints) - 1

        if n_segments <= 0:
            raise ValueError("Need at least 2 high-symmetry points to define a path")

        # Pre-allocate array for segment lengths
        seglen = np.zeros(n_segments)

        for i in range(n_segments):
            df = self.HSPoints[i + 1] - self.HSPoints[i]
            dk = df @ self.bvecs  # convert to reciprocal-space Cartesian coords
            seglen[i] = np.linalg.norm(dk)

        minlen = np.min(seglen)

        if minlen == 0:
            raise ValueError("Found zero-length segment in k-path")

        # Calculate divisions proportional to segment lengths
        divisions = seglen / minlen * self.ndivsm
        divisions = np.rint(divisions).astype(int)

        return divisions

    def print_qe_path(self, filename: str | Path | None = None) -> None:
        """
        Print or write the QE-style k-path specification.

        Generates a Quantum ESPRESSO style k-path file section with the format:
        - First line: number of high-symmetry points
        - Following lines: kx ky kz ndiv (fractional coords and divisions)

        Parameters
        ----------
        filename : str | Path | None, optional
            If provided, append to this file. Otherwise, print to stdout.

        Examples
        --------
        >>> kpath.print_qe_path()  # Print to console
        >>> kpath.print_qe_path("kpath.txt")  # Append to file
        """
        divisions = self.get_divisions()

        # QE convention: append 1 to divisions array
        divisions_qe = np.concatenate((divisions, [1]))

        lines = [f"{len(self.HSPoints)}"]
        for k, d in zip(self.HSPoints, divisions_qe):
            lines.append(f"{k[0]:.6f}  {k[1]:.6f}  {k[2]:.6f}  {d}")

        text = "\n".join(lines)

        if filename:
            filename = Path(filename)
            with open(filename, "a") as f:
                f.write(text + "\n")
        else:
            print(text)
    
    def get_kpoints_list(self, cart_coords: bool = False) -> list:
        """
        Get the list of k-points along the path with the specified divisions.
        """
        kpoints = []
        divisions = self.get_divisions()
        hs_points = self.HSPoints
        if cart_coords:
            hs_points = hs_points @ self.bvecs

        for i in range(len(hs_points) - 1):
            start = hs_points[i]
            end = hs_points[i + 1]
            ndiv = divisions[i]

            for j in range(ndiv):
                k = start + (end - start) * j / ndiv
                kpoints.append(k.tolist())
        # Append the last high-symmetry point
        kpoints.append(hs_points[-1].tolist())
        return kpoints

def read_eig_hpro(filename: str | Path) -> ndarray:
    """
    Read eigenvalues from a file in the hpro format.

    Parses eigenvalue data from hpro format files, extracting band energies
    and handling overlapping band indices.

    Parameters
    ----------
    filename : str | Path
        Path to the hpro format file

    Returns
    -------
    ndarray
        Array of eigenvalues with shape (n_bands, n_kpoints), starting from
        the maximum overlap index

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file format is invalid

    Examples
    --------
    >>> eigenvalues = read_eig_hpro("eigenvalues.hpro")
    >>> print(f"Shape: {eigenvalues.shape}")
    """
    filename = _validate_file_exists(filename, "filename")

    with open(filename, "r") as file:
        lines = file.readlines()

    if len(lines) < 3:
        raise ValueError(f"Invalid hpro file format: too few lines in {filename}")

    # Parse number of bands from third line
    try:
        nbnd = int(lines[2].split()[1])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Cannot parse number of bands from line 3: {e}")

    # Pre-allocate list for bands
    bands = [[] for _ in range(nbnd)]
    max_ov = 0

    # Parse eigenvalues
    for line in lines[3:]:
        parts = line.split()

        if len(parts) == 3:
            # Format: index band_index eigenvalue
            try:
                band_idx = int(parts[1]) - 1  # Convert to 0-indexed
                eigenval = float(parts[2])

                if 0 <= band_idx < nbnd:
                    bands[band_idx].append(eigenval)
            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        elif len(parts) == 2:
            # Format with overlap indicator
            try:
                # Extract overlap index from first character of second part
                overlap_idx = int(parts[1][0])
                max_ov = max(max_ov, overlap_idx)
            except (ValueError, IndexError):
                # Skip if unable to parse
                continue

    # Convert to numpy array, starting from max_ov
    result = np.array(bands[max_ov:])

    if result.size == 0:
        raise ValueError(f"No valid eigenvalue data found in {filename}")

    return result


def write_dh_structure(structure: Structure, save_dir: str | Path = "./") -> None:
    """
    Write structure data in DeepH format.

    Writes structure information to multiple files in the DeepH format:
    - element.dat: atomic species symbols
    - lat.dat: lattice vectors (transposed)
    - site_positions.dat: Cartesian coordinates (transposed)
    - info.json: metadata (spin, Fermi level)

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object to write
    save_dir : str | Path, optional
        Directory where files will be saved. Default is current directory.
        Will be created if it doesn't exist.

    Raises
    ------
    ValueError
        If structure is invalid

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> structure = Structure.from_file("POSCAR")
    >>> write_dh_structure(structure, "dh_output")
    """
    # Validate inputs
    _validate_structure(structure, "structure")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Write element.dat
    element_file = save_dir / "element.dat"
    with open(element_file, "w") as file:
        for species in structure.species:
            file.write(f"{species.symbol}\n")

    # Write lat.dat (transposed lattice matrix)
    lattice = np.transpose(structure.lattice.matrix)
    lat_file = save_dir / "lat.dat"
    with open(lat_file, "w") as file:
        for vec in lattice:
            file.write(f"{vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}\n")

    # Write site_positions.dat (transposed Cartesian coordinates)
    cart_coords = np.transpose(structure.cart_coords)
    pos_file = save_dir / "site_positions.dat"
    with open(pos_file, "w") as file:
        for coords in cart_coords:
            file.write(f"{coords[0]:.10f}  {coords[1]:.10f}  {coords[2]:.10f}\n")

    # Write info.json
    metadata = {"isspinful": False, "fermi_level": 0.0}
    info_file = save_dir / "info.json"
    with open(info_file, "w") as json_file:
        json.dump(metadata, json_file, indent=4)


@job
def cp_ion(outdirs: str | Path | list[str] | list[Path], ion_dir: str | Path) -> Path:
    """
    Copy ion files from output directories to a target directory.

    Searches for ion files in the specified output directories and copies
    them to a central ion directory. Used in jobflow workflows.

    Parameters
    ----------
    outdirs : str | Path | list[str] | list[Path]
        Single directory or list of directories containing ion files
    ion_dir : str | Path
        Target directory where ion files will be copied

    Returns
    -------
    Path
        Path to the ion directory

    Raises
    ------
    FileNotFoundError
        If source directories don't exist

    Examples
    --------
    >>> ion_path = cp_ion("calculation_output", "ions")
    >>> ion_path = cp_ion(["calc1", "calc2", "calc3"], "all_ions")
    """
    # Regex pattern to match element-specific ion files
    element_pattern = re.compile(r'\w+\.')

    # Normalize inputs to list of Path objects
    if isinstance(outdirs, (str, Path)):
        outdirs = [Path(outdirs)]
    else:
        outdirs = [Path(outdir) for outdir in outdirs]

    ion_dir = Path(ion_dir)
    ion_dir.mkdir(parents=True, exist_ok=True)

    # Process each output directory
    for outdir in outdirs:
        if not outdir.exists():
            raise FileNotFoundError(f"Output directory not found: {outdir}")

        # Find all ion files
        ion_files = list(outdir.glob('*ion'))

        # Filter for element-specific ion files (exactly one element prefix)
        element_ion_files = [
            ionf for ionf in ion_files
            if len(element_pattern.findall(ionf.name)) == 1
        ]

        # Copy filtered ion files
        for ionf in element_ion_files:
            copy(src=ionf, dst=ion_dir)

    return ion_dir

def read_pdos(filename:str) -> tuple:
    """
    Function to read the PDOS from a projwfc.x

    Args:
        filename (str): path to the projwfc.x output file
    Returns:
        tuple(np.ndarray,np.ndarray): energies and PDOS array
    """
    with open(filename,'r') as f:
        lines = f.readlines()
    orbital_type_number = len(lines[1].split())-2
    pdos = [[] for _ in range(orbital_type_number)]
    energies = []
    for line in lines:
        if '#' not in line:
            data = line.split()[-(orbital_type_number):]
            energies.append(float(line.split()[0]))
            for i in range(orbital_type_number):
                pdos[i].append(float(data[i]))
    return np.array(energies), np.array(pdos)
