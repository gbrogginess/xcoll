# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import re

import numpy as np

import xtrack as xt
from xtrack.particles import pdg


_FORMULA_TOKEN = re.compile(r'([A-Z][a-z]?)(\d*)')


def parse_molecule(formula):
    """
    Return the atomic composition of a molecule from its chemical formula.

    Parameters
    ----------
    formula : str
        Chemical formula without brackets or charges, e.g. ``'H2'``,
        ``'CH4'``, ``'H2O'``, ``'CO2'`` or ``'Ar'``.

    Returns
    -------
    composition : dict
        Mapping ``{element_symbol: number_of_atoms}``.
    """
    tokens = _FORMULA_TOKEN.findall(formula)
    if not tokens or ''.join(ee + nn for ee, nn in tokens) != formula:
        raise ValueError(f"Cannot parse the chemical formula {formula!r}.")
    composition = {}
    for element, count in tokens:
        # Raises for an unknown element symbol
        pdg.get_Z_from_element_name(element)
        composition[element] = composition.get(element, 0) + int(count or 1)
    return composition


def molecular_to_atomic_density(molecular_density):
    """
    Convert a molecular gas-density profile into atomic densities.

    The beam-gas studies take the density of each chemical element, since
    the interactions are computed per atom. This helper sums the
    contributions of all the molecules, e.g. ``H = 2 n_H2 + 4 n_CH4 +
    2 n_H2O``.

    Parameters
    ----------
    molecular_density : xtrack.Table or dict
        Profile with a column ``s`` [m] and one column per molecule, named
        after its chemical formula (``'H2'``, ``'CH4'``, ``'CO'``, ...), with
        the molecular density [molecules/m^3]. A ``name`` column, if present,
        is kept.

    Returns
    -------
    atomic_density : xtrack.Table
        Profile with the columns ``name`` (generated if absent) and ``s``,
        and one column per chemical element with the atomic density
        [atoms/m^3].
    """
    if isinstance(molecular_density, xt.Table):
        columns = list(molecular_density._col_names)
        get = molecular_density.__getitem__
    else:
        columns = list(molecular_density.keys())
        get = molecular_density.__getitem__
    if 's' not in columns:
        raise ValueError("The molecular density must have an `s` column.")

    data = {}
    s = np.asarray(get('s'), dtype=float)
    if 'name' in columns:
        data['name'] = np.asarray(get('name'))
    else:
        data['name'] = np.array([f'gas.{ii}' for ii in range(s.size)])
    data['s'] = s
    atomic = {}
    for molecule in columns:
        if molecule in ('s', 'name'):
            continue
        density = np.asarray(get(molecule), dtype=float)
        for element, count in parse_molecule(molecule).items():
            atomic[element] = atomic.get(element, 0.0) + count*density
    if not atomic:
        raise ValueError("The molecular density has no molecule column.")
    data.update(atomic)
    return xt.Table(data)
