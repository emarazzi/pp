"""
Band structure evaluation and comparison utilities.

This module provides functions for comparing band structures, computing band gaps,
and calculating weighted band distances using Fermi-Dirac distributions.
"""

import numpy as np
from numpy import ndarray


def _validate_bands_input(bands: list | ndarray, param_name: str = "bands") -> ndarray:
    """
    Validate and convert bands input to numpy array.

    Parameters
    ----------
    bands : list | ndarray
        Band structure data
    param_name : str
        Parameter name for error messages

    Returns
    -------
    ndarray
        Validated numpy array

    Raises
    ------
    ValueError
        If input is empty, contains NaN, or has invalid shape
    """
    bands_array = np.array(bands)

    if bands_array.size == 0:
        raise ValueError(f"{param_name} cannot be empty")

    if np.any(np.isnan(bands_array)):
        raise ValueError(f"{param_name} contains NaN values")

    if bands_array.ndim < 1:
        raise ValueError(f"{param_name} must be at least 1-dimensional")

    return bands_array


def band_comparison(band1: list | ndarray, band2: list | ndarray) -> float:
    """
    Compare two band structures using root mean square error.

    Computes the RMSE between two band structures, normalized by the number of
    k-points and the minimum number of bands.

    Parameters
    ----------
    band1 : list | ndarray
        First band structure with shape (n_bands, n_kpoints)
    band2 : list | ndarray
        Second band structure with shape (m_bands, n_kpoints)

    Returns
    -------
    float
        Total RMSE across all bands

    Raises
    ------
    ValueError
        If the bands have different numbers of k-points

    Examples
    --------
    >>> band1 = np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]])
    >>> band2 = np.array([[0.01, 0.11, 0.21], [1.01, 1.11, 1.21]])
    >>> total_rmse = band_comparison(band1, band2)
    """
    band1 = _validate_bands_input(band1, "band1")
    band2 = _validate_bands_input(band2, "band2")

    if band1.shape[1] != band2.shape[1]:
        raise ValueError(
            f"Bands have different numbers of k-points: "
            f"{band1.shape[1]} vs {band2.shape[1]}"
        )

    # Determine minimum number of bands to compare
    n_bands_min = min(band1.shape[0], band2.shape[0])
    n_kpoints = band1.shape[1]

    # Vectorized computation: compute squared differences for all bands at once
    diff_squared = np.abs(band1[:n_bands_min] - band2[:n_bands_min]) ** 2

    # Total mean squared error
    total_mse = np.sum(diff_squared) / (n_kpoints * n_bands_min)

    return np.sqrt(total_mse)


def shift_vbm(bands: list | ndarray, fermie: float) -> ndarray:
    """
    Shift band energies so the valence band maximum (VBM) is at zero.

    Finds the highest energy band below the Fermi level and shifts all bands
    so that this energy becomes zero. Returns a copy of the shifted bands.

    Parameters
    ----------
    bands : list | ndarray
        Band energies with shape (n_bands, n_kpoints) or (n_kpoints,)
    fermie : float
        Fermi energy level

    Returns
    -------
    ndarray
        Shifted band energies (copy of input)

    Raises
    ------
    ValueError
        If no bands exist below the Fermi level

    Examples
    --------
    >>> bands = np.array([[-1.0, -0.9], [0.5, 0.6], [2.0, 2.1]])
    >>> shifted = shift_vbm(bands, fermie=1.0)
    """
    bands_array = _validate_bands_input(bands, "bands")

    below_fermi_mask = bands_array < fermie

    if not np.any(below_fermi_mask):
        raise ValueError(
            f"No bands found below Fermi level ({fermie}). "
            f"Band range: [{np.min(bands_array):.3f}, {np.max(bands_array):.3f}]"
        )

    top_vb = np.max(bands_array[below_fermi_mask])

    # Return shifted copy to preserve immutability
    return bands_array - top_vb


def shift_cbm(bands: list | ndarray, fermie: float) -> ndarray:
    """
    Shift band energies so the conduction band minimum (CBM) is at zero.

    Finds the lowest energy band above the Fermi level and shifts all bands
    so that this energy becomes zero. Returns a copy of the shifted bands.

    Parameters
    ----------
    bands : list | ndarray
        Band energies with shape (n_bands, n_kpoints) or (n_kpoints,)
    fermie : float
        Fermi energy level

    Returns
    -------
    ndarray
        Shifted band energies (copy of input)

    Raises
    ------
    ValueError
        If no bands exist above the Fermi level

    Examples
    --------
    >>> bands = np.array([[-1.0, -0.9], [0.5, 0.6], [2.0, 2.1]])
    >>> shifted = shift_cbm(bands, fermie=0.0)
    """
    bands_array = _validate_bands_input(bands, "bands")

    above_fermi_mask = bands_array > fermie

    if not np.any(above_fermi_mask):
        raise ValueError(
            f"No bands found above Fermi level ({fermie}). "
            f"Band range: [{np.min(bands_array):.3f}, {np.max(bands_array):.3f}]"
        )

    bottom_cb = np.min(bands_array[above_fermi_mask])

    # Return shifted copy to preserve immutability
    return bands_array - bottom_cb


def get_bandgap(bands: list | ndarray, fermie: float) -> float:
    """
    Calculate the band gap energy.

    Computes the difference between the conduction band minimum (lowest energy
    above Fermi level) and the valence band maximum (highest energy below Fermi level).

    Parameters
    ----------
    bands : list | ndarray
        Band energies with shape (n_bands, n_kpoints) or (n_kpoints,)
    fermie : float
        Fermi energy level

    Returns
    -------
    float
        Band gap energy (always positive)

    Raises
    ------
    ValueError
        If no bands exist above or below the Fermi level

    Examples
    --------
    >>> bands = np.array([[-1.0, -0.9], [0.5, 0.6], [2.0, 2.1]])
    >>> gap = get_bandgap(bands, fermie=0.0)
    >>> print(f"Band gap: {gap:.3f} eV")
    """
    bands_array = _validate_bands_input(bands, "bands")

    below_fermi_mask = bands_array < fermie
    above_fermi_mask = bands_array > fermie

    if not np.any(below_fermi_mask):
        raise ValueError(
            f"No bands found below Fermi level ({fermie}). "
            f"Band range: [{np.min(bands_array):.3f}, {np.max(bands_array):.3f}]"
        )

    if not np.any(above_fermi_mask):
        raise ValueError(
            f"No bands found above Fermi level ({fermie}). "
            f"Band range: [{np.min(bands_array):.3f}, {np.max(bands_array):.3f}]"
        )

    top_vb = np.max(bands_array[below_fermi_mask])
    bottom_cb = np.min(bands_array[above_fermi_mask])

    return np.abs(bottom_cb - top_vb)


def fd(e: float | ndarray, fermie: float, nu: float, sigma: float) -> float | ndarray:
    """
    Compute the Fermi-Dirac distribution.

    Parameters
    ----------
    e : float | ndarray
        Energy values
    fermie : float
        Fermi energy level
    nu : float
        Chemical potential shift
    sigma : float
        Temperature broadening parameter (k_B * T)

    Returns
    -------
    float | ndarray
        Fermi-Dirac occupation values in range [0, 1]

    Examples
    --------
    >>> occupation = fd(e=0.0, fermie=0.0, nu=0.0, sigma=0.025)
    """
    return 1 / (np.exp((e - fermie - nu) / sigma) + 1)


def fdtilde(ref: ndarray, new: ndarray, fermie: float, nu: float, sigma: float) -> ndarray:
    """
    Compute geometric mean of Fermi-Dirac distributions for two band structures.

    This weighting function emphasizes regions where both bands are occupied
    or both are unoccupied, used in weighted band distance calculations.

    Parameters
    ----------
    ref : ndarray
        Reference band energies
    new : ndarray
        New band energies
    fermie : float
        Fermi energy level
    nu : float
        Chemical potential shift
    sigma : float
        Temperature broadening parameter

    Returns
    -------
    ndarray
        Geometric mean of Fermi-Dirac distributions

    Examples
    --------
    >>> ref_band = np.array([0.0, 0.1, 0.2])
    >>> new_band = np.array([0.01, 0.11, 0.21])
    >>> weights = fdtilde(ref_band, new_band, fermie=0.0, nu=0.0, sigma=0.025)
    """
    # Compute FD distributions once and cache
    fd_ref = fd(ref, fermie, nu, sigma)
    fd_new = fd(new, fermie, nu, sigma)

    return np.sqrt(fd_ref * fd_new)


def band_distance(ref: ndarray, new: ndarray, fermie: float, nu: float, sigma: float) -> float:
    """
    Calculate weighted band distance using Fermi-Dirac weighting.

    Computes a distance metric between two band structures that emphasizes
    differences near the Fermi level using Fermi-Dirac distribution weighting.

    Parameters
    ----------
    ref : ndarray
        Reference band structure with shape (n_bands, n_kpoints)
    new : ndarray
        New band structure with shape (m_bands, n_kpoints)
    fermie : float
        Fermi energy level
    nu : float
        Chemical potential shift
    sigma : float
        Temperature broadening parameter

    Returns
    -------
    float
        Weighted band distance metric

    Raises
    ------
    RuntimeError
        If bands have different numbers of k-points

    Examples
    --------
    >>> ref = np.array([[0.0, 0.1], [1.0, 1.1]])
    >>> new = np.array([[0.01, 0.11], [1.01, 1.11]])
    >>> dist = band_distance(ref, new, fermie=0.5, nu=0.0, sigma=0.025)
    """
    if ref.shape[1] != new.shape[1]:
        raise RuntimeError(
            f"Bands have different numbers of k-points: "
            f"{ref.shape[1]} vs {new.shape[1]}"
        )

    n_bands_min = min(ref.shape[0], new.shape[0])
    eta = 0.0

    # Vectorize the main computation over bands
    for i in range(n_bands_min):
        # Compute weighting function
        weight = fdtilde(ref[i], new[i], fermie, nu, sigma)

        # Compute weighted squared difference
        diff_squared = (ref[i] - new[i]) ** 2
        weighted_sum = np.sum(weight * diff_squared)
        weight_sum = np.sum(weight)

        # Avoid division by zero
        if weight_sum > 0:
            eta += np.sqrt(weighted_sum / weight_sum)

    return eta


def band_folding(
        bands: list|ndarray,
        supercell_size:list,
        reciprocal_lattice:list|ndarray
        ) -> ndarray:
    """
    Fold band structure into the supercell Brillouin zone.

    Parameters
    ----------
    bands : list | ndarray
        Original band structure with shape (n_bands, n_kpoints)
    supercell_size : list
        Supercell size as a list of integers [n1, n2, n3]
    reciprocal_lattice : list | ndarray
        Reciprocal lattice vectors as a 3x3 array
    
    Returns
    -------
    ndarray
        Folded band structure with shape (n_bands, n_kpoints_supercell)
    """
    bands_array = _validate_bands_input(bands, "bands")


def calculate_shift(bands_a,bands_b) -> float:
    """
    Function to calculate the optimal shift between two bands
    to align their centers.

    Parameters
    ----------
    bands_a : list | ndarray
        First band structure
    bands_b : list | ndarray
        Second band structure

    Returns
    -------
    float
        Optimal shift value to align the centers of the two bands
    """
    bands_a = np.array(bands_a)
    bands_b = np.array(bands_b)
    center_a = np.mean(bands_a)
    center_b = np.mean(bands_b)

    return center_b - center_a
