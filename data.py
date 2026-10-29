from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .features import density_features
from .descriptor import transform_rank4
from . import preprocessing
from .registry import MOLECULES, DEFAULT_DESCRIPTOR_DIR, nh3_selected_aos

DEFAULT_INPUT = DEFAULT_DESCRIPTOR_DIR


@dataclass
class Sample:
    sample_id: str
    molecule: str
    geometry_index: int
    scan_coordinate: float
    bond_length: float
    active_ao_indices: np.ndarray
    active_ao_labels: tuple[str, ...]
    source_selected_ao_indices: np.ndarray
    pii: np.ndarray
    fii: np.ndarray
    pij: np.ndarray
    fij: np.ndarray
    target_energy: float
    rhf_total_energy: float
    fci_total_energy: float
    source_file: Path
    descriptors: dict[str, np.ndarray]


@dataclass
class Constants:
    molecule: str
    training_sample_ids: tuple[str, ...]
    n_active: int
    Pii_mean: float = 0.0
    Pii_std: float = 1.0
    Fii_mean: float = 0.0
    Fii_std: float = 1.0
    Pij_abs95: float = 0.0
    alpha: float = 0.0
    mu: tuple[float, ...] | None = None
    sigma: tuple[float, ...] | None = None
    sigma_safe: tuple[float, ...] | None = None
    constant_mask: tuple[bool, ...] | None = None


def density_descriptors(P, gamma_raw, selected):
    values = density_features(P)
    i, j = np.triu_indices(len(P), k=1)
    I, J = selected[i], selected[j]
    values['Gamma_2'] = gamma_raw[I, I, J, J]
    return values


def load_sample(path, molecule, geometry_index, sample_id=None, *, include_reference=True, include_f=True):
    symbols, active = MOLECULES[molecule]
    path = Path(path)
    with np.load(path, allow_pickle=False) as raw:
        if 'Gamma_MP2_AO_pyscf' not in raw:
            raise ValueError(f'{path}: missing native PySCF MP2 tensor Gamma_MP2_AO_pyscf; '
                             'generate native MP2 descriptors in a new directory')
        atoms = []
        for symbol in symbols:
            atoms.append(next(i for i, s in enumerate(raw['symbols'])
                              if s == symbol and i not in atoms))
        labels = [str(label).split() for label in raw['ao_labels']]
        full_order = [i for atom in atoms for i, label in enumerate(labels)
                      if int(label[0]) == atom]
        selected = np.asarray(nh3_selected_aos(raw['symbols'], raw['ao_labels'])
                              if molecule == 'NH3' else [full_order[i] for i in active])
        ao_labels = tuple(f'{atoms.index(int(labels[i][0]))} {labels[i][1]} {labels[i][2]}'
                          for i in selected)

        P = raw['P_L'][np.ix_(selected, selected)]
        F = raw['F_L'][np.ix_(selected, selected)] if include_f else None
        gamma = transform_rank4(raw['Gamma_MP2_AO_pyscf'], raw['S_half'])
        rhf, fci = ((float(raw['E_RHF_input']), float(raw['E_FCI']))
                    if include_reference else (np.nan, np.nan))
        return Sample(
            sample_id=f'{molecule}/{geometry_index:03d}' if sample_id is None else sample_id,
            molecule=molecule,
            geometry_index=geometry_index, scan_coordinate=float(raw['q_A']),
            bond_length=float(raw['bond_length_A']), active_ao_indices=np.asarray(active),
            active_ao_labels=ao_labels, source_selected_ao_indices=selected,
            pii=np.diag(P), fii=np.diag(F) if F is not None else None, pij=P, fij=F,
            target_energy=fci - rhf, rhf_total_energy=rhf, fci_total_energy=fci,
            source_file=path,
            descriptors=density_descriptors(P, gamma, selected),
        )


def load_data(input_dir=DEFAULT_INPUT, molecule='CO', *, include_f=True, geometric=False):
    if geometric:
        from .FG import load_data as load_fg_data
        return load_fg_data(input_dir, molecule)
    samples = [load_sample(Path(input_dir) / molecule / f'{index:03d}.npz', molecule, index, include_f=include_f)
               for index in range(1, 31)]
    return split_samples(samples)


def split_samples(samples):
    samples.sort(key=lambda s: (s.scan_coordinate, s.geometry_index))
    training = tuple(s for s in samples if s.geometry_index % 2 == 1)
    testing = tuple(s for s in samples if s.geometry_index % 2 == 0)
    return training, testing


def fit_constants(training, name='MB1'):
    if name == 'FG':
        from .FG import fit_constants as fit_fg_constants
        return fit_fg_constants(training)
    if name in preprocessing.MANY_BODY_MODELS:
        return preprocessing.fit_many_body_constants(training, name)
    if name != 'MB1':
        raise ValueError(f'Unknown paper_source_v5 model: {name}')
    ids = tuple(s.sample_id for s in training)
    Pii = np.concatenate([s.pii for s in training])
    return Constants(training[0].molecule, ids, len(training[0].pii),
                     Pii_mean=float(np.mean(Pii)), Pii_std=float(np.std(Pii, ddof=0)))


def constants_from_saved(saved, name):
    if name == 'FG':
        from .FG import constants_from_saved as fg_constants_from_saved
        return fg_constants_from_saved(saved)
    if name in preprocessing.MANY_BODY_MODELS:
        return preprocessing.constants_from_saved(saved, name)
    if name != 'MB1':
        raise KeyError(name)
    if set(saved) != {field.name for field in fields(Constants)}:
        raise ValueError('Unexpected preprocessing schema')
    return Constants(**saved)
