"""Stored full-AO data, valence descriptors, and training-set standardization."""
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT / 'results' / 'descriptor'

# Canonical atom order and valence AO indices order
MOLECULES = {
    'LiH': (('Li', 'H'), (1, 2, 3, 4, 5)),
    'BeH2': (('H', 'Be', 'H'), (0, 2, 3, 4, 5, 6)),
    'H2O': (('O', 'H', 'H'), (1, 2, 3, 4, 5, 6)),
    'NH3': (('N', 'H', 'H', 'H'), (1, 2, 3, 4, 5, 6, 7)),
    'N2': (('N', 'N'), (1, 2, 3, 4, 6, 7, 8, 9)),
    'CO': (('C', 'O'), (1, 2, 3, 4, 6, 7, 8, 9)),
    'HF': (('H', 'F'), (0, 2, 3, 4, 5)),
    'H2S': (('S', 'H', 'H'), (2, 6, 7, 8, 9, 10)),
    'H2O2': (('O', 'O', 'H', 'H'), (1, 2, 3, 4, 6, 7, 8, 9, 10, 11)),
}


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
    fingerprint: np.ndarray
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


def density_descriptors(P, gamma_raw, lambda_raw, selected):
    i, j = np.triu_indices(len(P), k=1)
    D = P[i, i] * P[j, j]
    O = P[i, j] * P[j, i]
    I, J = selected[i], selected[j]
    # Raw G[p,q,r,s] = Methods[p,r,q,s]; Methods[I,J,I,J] = G[I,I,J,J].
    values = dict(D=D, O=O, O_square=P[i, j] * P[i, j],
                  Gamma0=D - 0.5 * O, Gamma=gamma_raw[I, I, J, J],
                  Lambda=lambda_raw[I, I, J, J])
    values['Gamma0_2'] = values['Gamma0']
    i, j, k = np.asarray(list(combinations(range(len(P)), 3))).T
    A = P[i, i] * P[j, j] * P[k, k]
    B = P[i, i] * P[j, k] ** 2  
    C = P[i, j] * P[i, k] * P[j, k]
    values.update(A=A, B=B, C=C,
                  Gamma0_3=A - 0.5 * (B + P[j, j] * P[i, k] ** 2
                                     + P[k, k] * P[i, j] ** 2) + 0.5 * C)
    indices = np.asarray(list(combinations(range(len(P)), 4)))
    i, j, k, l = indices.T
    values.update(
        Q_A=P[i, i] * P[j, j] * P[k, k] * P[l, l],
        Q_B=P[i, i] * P[j, j] * P[k, l] * P[k, l],
        Q_C=P[i, j] * P[i, j] * P[k, l] * P[k, l],
        Q_D=P[i, j] * P[i, k] * P[j, k] * P[l, l],
        Q_E=P[i, j] * P[i, k] * P[j, l] * P[k, l],
    )
    # Spin-summed P = 2*gamma_spin; sign(pi)*2**(cycles(pi)-4).
    gamma4 = np.zeros(len(indices), dtype=np.float64)
    for permutation in permutations(range(4)):
        inversions = sum(permutation[i] > permutation[j]
                         for i in range(4) for j in range(i + 1, 4))
        seen, cycles = set(), 0
        for i in range(4):
            if i not in seen:
                cycles += 1
                while i not in seen:
                    seen.add(i)
                    i = permutation[i]
        term = np.ones(len(indices), dtype=np.float64)
        for j in range(4):
            term *= P[indices[:, j], indices[:, permutation[j]]]
        gamma4 += (-1) ** inversions * 2.0 ** (cycles - 4) * term
    values['Gamma0_4'] = gamma4
    return values


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def angle(a, center, b):
    u, v = np.asarray(a) - np.asarray(center), np.asarray(b) - np.asarray(center)
    norms = float(np.linalg.norm(u)) * float(np.linalg.norm(v))
    return float(np.arccos(np.clip(float(np.dot(u, v) / norms), -1.0, 1.0)))


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = -(p1 - p0), p2 - p1, p3 - p2
    axis = b1 / float(np.linalg.norm(b1))
    u = b0 - float(np.dot(b0, axis)) * axis
    v = b2 - float(np.dot(b2, axis)) * axis
    return float(np.arctan2(float(np.dot(np.cross(axis, u), v)), float(np.dot(u, v))))


def fg_features(molecule, coordinates):
    c = coordinates
    if molecule in ('LiH', 'N2', 'CO', 'HF'):
        d = np.asarray([distance(*c)])
        return np.concatenate((d, 1.0 / d))
    if molecule in ('BeH2', 'H2O', 'H2S'):
        x, h1, h2 = c
        d1, d2, dhh = distance(x, h1), distance(x, h2), distance(h1, h2)
        return np.concatenate(([d1, d2, dhh, 1.0 / d1, 1.0 / d2], [angle(h1, x, h2)]))
    if molecule == 'NH3':
        x, h1, h2, h3 = c
        d = np.asarray([distance(x, h) for h in (h1, h2, h3)])
        return np.concatenate((d, 1.0 / d, [angle(h1, x, h2)]))
    o1, o2, h1, h2 = c
    dhh = distance(h1, h2)
    d = [distance(o1, o2), distance(o1, h1), distance(o2, h2),
         distance(o1, h2), distance(o2, h1), dhh, 1.0 / dhh]
    return np.concatenate((d, [angle(o2, o1, h1), angle(o1, o2, h2),
                               dihedral(h1, o1, o2, h2)]))


def load_data(input_dir=DEFAULT_INPUT, molecule='CO'):
    symbols, active = MOLECULES[molecule]
    samples = []
    for geometry_index in range(1, 31):
        path = Path(input_dir) / molecule / f'{geometry_index:03d}.npz'
        with np.load(path, allow_pickle=False) as raw:
            atoms = []
            for symbol in symbols:
                atoms.append(next(i for i, s in enumerate(raw['symbols'])
                                  if s == symbol and i not in atoms))
            labels = [str(label).split() for label in raw['ao_labels']]
            full_order = [i for atom in atoms for i, label in enumerate(labels)
                          if int(label[0]) == atom]
            selected = np.asarray([full_order[i] for i in active])
            ao_labels = tuple(f'{atoms.index(int(labels[i][0]))} {labels[i][1]} {labels[i][2]}'
                              for i in selected)
            
            fg_order = [atoms[i] for i in (1, 2, 0)] if molecule == 'BeH2' else atoms
            P = raw['P_L'][np.ix_(selected, selected)]
            F = raw['F_L'][np.ix_(selected, selected)]
            half = raw['S_half']
            # Transform the total MP2 2-RDM in full AO space before valence selection.
            gamma = np.einsum('ap,bq,cr,ds,pqrs->abcd', half, half, half, half,
                              raw['Gamma_MP2_AO_consistent'], optimize=True)
            cumulant = raw['Lambda_L']  # Gamma_MP2 - Gamma0[P_RHF], in full Lowdin space.
            rhf, fci = float(raw['E_RHF_input']), float(raw['E_FCI'])
            samples.append(Sample(
                sample_id=f'{molecule}/{geometry_index:03d}', molecule=molecule,
                geometry_index=geometry_index, scan_coordinate=float(raw['q_A']),
                bond_length=float(raw['bond_length_A']), active_ao_indices=np.asarray(active),
                active_ao_labels=ao_labels, source_selected_ao_indices=selected,
                pii=np.diag(P), fii=np.diag(F), pij=P, fij=F,
                target_energy=fci - rhf, rhf_total_energy=rhf, fci_total_energy=fci,
                source_file=path, fingerprint=fg_features(molecule, raw['cartesian_A'][fg_order]),
                descriptors=density_descriptors(P, gamma, cumulant, selected),
            ))
    samples.sort(key=lambda s: (s.scan_coordinate, s.geometry_index))
    if molecule == 'H2O2':
        torsion = np.asarray([s.fingerprint[-1] for s in samples])
        if np.any(np.abs(np.diff(torsion)) > np.pi):
            for sample, value in zip(samples, np.unwrap(torsion)):
                sample.fingerprint[-1] = value
    training = tuple(s for s in samples if s.geometry_index % 2 == 1)
    testing = tuple(s for s in samples if s.geometry_index % 2 == 0)
    return training, testing


def fit_constants(training, name='MB1'):
    ids = tuple(s.sample_id for s in training)
    molecule = training[0].molecule
    if name == 'FG':
        X = np.vstack([s.fingerprint for s in training])
        mu, sigma = np.mean(X, axis=0), np.std(X, axis=0, ddof=0)
        mask = sigma < 1e-12
        safe = sigma.copy()
        safe[mask] = 1.0
        return Constants(molecule, ids, X.shape[1], mu=tuple(mu.tolist()),
                         sigma=tuple(sigma.tolist()), sigma_safe=tuple(safe.tolist()),
                         constant_mask=tuple(mask.tolist()))
    Pii = np.concatenate([s.pii for s in training])
    Fii = np.concatenate([s.fii for s in training])
    n = len(training[0].pii)
    i, j = np.triu_indices(n, k=1)
    Pij = np.concatenate([s.pij[i, j] for s in training])
    pair95 = float(np.percentile(np.abs(Pij), 95, method='linear'))
    return Constants(molecule, ids, n, float(np.mean(Pii)), float(np.std(Pii, ddof=0)),
                     float(np.mean(Fii)), float(np.std(Fii, ddof=0)),
                     pair95, float(np.arctanh(0.9)) / pair95)
