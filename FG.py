"""Production geometric fingerprint baseline, as archived in FG_production.

Distances use Angstrom, inverse distances use Angstrom^-1, angles use radians.
One feature occupies one qubit, including columns constant on the training set.
"""
from dataclasses import dataclass, fields
import math
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from .data import Constants
from .models import hea, readout_features
from .registry import DEFAULT_DESCRIPTOR_DIR, MOLECULES

FEATURE_NAMES = {
    'HF': ('rHF', '1/rHF'),
    'N2': ('rNN', '1/rNN'),
    'CO': ('rCO', '1/rCO'),
    'H2O': ('rOH1', 'rOH2', 'rH1H2', '1/rOH1', '1/rOH2', 'angle(H1,O,H2)'),
    'NH3': ('rNH1', 'rNH2', 'rNH3', '1/rNH1', '1/rNH2', '1/rNH3', 'angle(H1,N,H2)'),
}
FEATURE_UNITS = {molecule: tuple('Angstrom^-1' if name.startswith('1/') else
                                'Angstrom' if name.startswith('r') else 'radian'
                                for name in names)
                 for molecule, names in FEATURE_NAMES.items()}


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def angle(a, center, b):
    u, v = np.asarray(a) - np.asarray(center), np.asarray(b) - np.asarray(center)
    norms = float(np.linalg.norm(u)) * float(np.linalg.norm(v))
    return float(np.arccos(np.clip(float(np.dot(u, v) / norms), -1.0, 1.0)))


def geometric_features(molecule, coordinates):
    c = np.asarray(coordinates, dtype=np.float64)
    if molecule in ('HF', 'N2', 'CO'):
        d = np.asarray([distance(*c)])
        return np.concatenate((d, 1.0 / d))
    if molecule == 'H2O':
        x, h1, h2 = c
        d1, d2, dhh = distance(x, h1), distance(x, h2), distance(h1, h2)
        return np.concatenate(([d1, d2, dhh, 1.0 / d1, 1.0 / d2], [angle(h1, x, h2)]))
    if molecule == 'NH3':
        x, h1, h2, h3 = c
        d = np.asarray([distance(x, h) for h in (h1, h2, h3)])
        return np.concatenate((d, 1.0 / d, [angle(h1, x, h2)]))
    raise ValueError(f'Unsupported FG molecule: {molecule}')


@dataclass
class Sample:
    sample_id: str
    molecule: str
    geometry_index: int
    scan_coordinate: float
    bond_length: float
    target_energy: float
    rhf_total_energy: float
    fci_total_energy: float
    source_file: Path
    fingerprint: np.ndarray


def load_sample(path, molecule, geometry_index, sample_id=None, *, include_reference=True, include_f=False):
    """Read geometry and energies only; FG does not require electronic descriptors."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as raw:
        atoms = []
        for symbol in MOLECULES[molecule][0]:
            atoms.append(next(i for i, s in enumerate(raw['symbols'])
                              if s == symbol and i not in atoms))
        rhf, fci = ((float(raw['E_RHF_input']), float(raw['E_FCI']))
                    if include_reference else (np.nan, np.nan))
        return Sample(f'{molecule}/{geometry_index:03d}' if sample_id is None else sample_id,
                      molecule, geometry_index, float(raw['q_A']), float(raw['bond_length_A']),
                      fci - rhf, rhf, fci, path,
                      geometric_features(molecule, raw['cartesian_A'][atoms]))


def prepare_samples(samples):
    return sorted(samples, key=lambda s: (s.scan_coordinate, s.geometry_index))


def load_data(input_dir=DEFAULT_DESCRIPTOR_DIR, molecule='CO', *, include_f=False):
    from .data import split_samples
    samples = [load_sample(Path(input_dir) / molecule / f'{i:03d}.npz', molecule, i)
               for i in range(1, 31)]
    return split_samples(prepare_samples(samples))


def fit_constants(training):
    X = np.vstack([s.fingerprint for s in training])
    mu, sigma = np.mean(X, axis=0), np.std(X, axis=0, ddof=0)
    mask = sigma < 1e-12
    safe = sigma.copy()
    safe[mask] = 1.0
    return Constants(training[0].molecule, tuple(s.sample_id for s in training), X.shape[1],
                     mu=tuple(mu.tolist()), sigma=tuple(sigma.tolist()),
                     sigma_safe=tuple(safe.tolist()), constant_mask=tuple(mask.tolist()))


def constants_from_saved(saved):
    if set(saved) != {f.name for f in fields(Constants)}:
        raise ValueError('Unexpected FG preprocessing schema')
    c = Constants(**saved)
    width = len(FEATURE_NAMES[c.molecule])
    mu, sigma, safe, mask = [np.asarray(getattr(c, k))
                            for k in ('mu', 'sigma', 'sigma_safe', 'constant_mask')]
    if (c.n_active != width or any(v.shape != (width,) for v in (mu, sigma, safe, mask))
            or not np.isfinite([mu, sigma, safe]).all() or np.any(sigma < 0)
            or not np.array_equal(mask, sigma < 1e-12)
            or not np.array_equal(safe, np.where(mask, 1.0, sigma))):
        raise ValueError('Invalid FG feature normalization or feature count')
    return c


def gate_angles(fingerprint, constants):
    p = math.pi / 2.0 * (fingerprint - np.asarray(constants.mu)) / np.asarray(constants.sigma_safe)
    p[np.asarray(constants.constant_mask, dtype=bool)] = 0.0
    return np.clip(p, -math.pi, math.pi)


def model_definition():
    return dict(pipeline_version='paper_source_v5',
                parameter_schema='paper_source_v5_fg_production_v1',
                model_parameter_schema='paper_source_v5_fg_production_v1:FG',
                canonical_model='FG', one_body_parameter_names=[], one_body_parameter_count=0,
                encoding_parameter_count=0, feature_schema='FG_production_full_geometry_v1',
                source_implementation='codes_v3.data, codes_v3.models, codes_v3.train; FG_production',
                preprocessing_definition='per-feature training mean/std (ddof=0); sigma<1e-12: safe=1 and angle=0; retain every column',
                encoding_definition='RY(clip((pi/2)*(x-mu)/sigma_safe,-pi,pi)); one qubit per feature',
                circuit_definition='two shared RY(phi) layers, each followed by ordered CNOT ring (n>1)',
                readout_definition='bias+dot(weights,[Z padded to max(8,n),mean(Z),mean(Z**2),mean(Z**3)])',
                initialization_definition='seeded float64 Gaussian std=0.05 in draws (2,),(),(max(8,n)+3,); overwrite bias with training target mean',
                optimizer_definition='Adam parameter views: phi, scalar bias, readout weights')


class FGModel:
    def __init__(self, constants, name='FG', samples=()):
        self.constants, self.name, self.n = constants, name, constants.n_active
        self.context_width = max(8, self.n)
        self.encoder_names, self.bias_index = (), 2
        self.names = ('phi0', 'phi1', 'bias') + tuple(
            f'Z{i}_weight' for i in range(self.context_width)) + (
            'mean_Z_weight', 'mean_Z2_weight', 'mean_Z3_weight')
        self.num_parameters = self.readout_end = len(self.names)
        self.parameter_indices = {key: i for i, key in enumerate(self.names)}
        prepared = [s if hasattr(s, 'fingerprint') else load_sample(
            s.source_file, s.molecule, s.geometry_index, s.sample_id, include_reference=False)
                    for s in samples]
        self.features = {s.sample_id: s.fingerprint for s in prepare_samples(prepared)}

        def circuit(p, phi):
            for i in range(self.n):
                qml.RY(p[i], wires=i)
            hea(self.n, phi)
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(self.n))

        self.circuit, self.qnode = circuit, None

    def quantum_arguments(self, sample, theta):
        if self.qnode is None:
            device = qml.device('default.qubit', wires=self.n)
            self.qnode = qml.QNode(self.circuit, device, interface='torch', diff_method='backprop')
        fingerprint = (sample.fingerprint if hasattr(sample, 'fingerprint')
                       else self.features[sample.sample_id])
        angles = torch.tensor(gate_angles(fingerprint, self.constants), dtype=theta.dtype, device=theta.device)
        return angles, theta[:2]

    def quantum_features(self, sample, theta):
        arguments = self.quantum_arguments(sample, theta)
        return torch.stack(self.qnode(*arguments)).reshape(-1)

    def raw_readout_features(self, sample, theta):
        return readout_features(self.quantum_features(sample, theta), self.context_width)

    def predict_batch(self, samples, theta):
        return torch.stack([self(sample, theta) for sample in samples])

    def __call__(self, sample, theta):
        return theta[2] + torch.dot(theta[3:], self.raw_readout_features(sample, theta))
