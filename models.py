from itertools import combinations
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache' / 'matplotlib'))

import numpy as np
import pennylane as qml
import torch

from .preprocessing import (
    MANY_BODY_MODELS, HIERARCHY_LAYERS, many_body_features,
    standardized_density_diagonal,
)


_PAIR = (('a2', 'D'), ('b2', 'O'))
_C3 = tuple(zip(('a3', 'b3', 'c3'), ('S111', 'S21', 'S3')))
_C4 = tuple(zip(('a4', 'b4', 'c4', 'd4', 'e4'),
                ('S1111', 'S211', 'S22', 'S31', 'S4')))
MODELS = {
    'MB1': (),
    'MB2': (_PAIR,),
    'MB3': (_PAIR, _C3),
    'MB4': (_PAIR, _C3, _C4),
    'MB2_gamm0': ((('a2', 'Gamma0_2'),),),
    'MB2_gamma': ((('a2', 'Gamma_2'),),),
    **HIERARCHY_LAYERS,
}
DEFAULT_MODELS = tuple(MODELS)
MODELS['FG'] = ()
TRAINABLE_MODELS = tuple(MODELS)


def hea(n, phi):
    for layer in range(2):
        for i in range(n):
            qml.RY(phi[layer], wires=i)
        if n > 1:
            for i in range(n):
                qml.CNOT(wires=[i, (i + 1) % n])


def readout_features(z, context_width):
    padded = torch.cat((z, z.new_zeros(context_width - z.numel())))
    moments = torch.stack((torch.mean(z), torch.mean(z ** 2), torch.mean(z ** 3)))
    return torch.cat((padded, moments))


class Model:
    def __new__(cls, constants, name, samples):
        if name == 'FG':
            from .FG import FGModel
            return FGModel(constants, name, samples)
        return super().__new__(cls)

    def __init__(self, constants, name, samples):
        if name not in MODELS:
            raise ValueError(f'Unknown paper_source_v5 model: {name}')
        self.constants = constants
        self.name = name
        self.n = constants.n_active
        self.context_width = max(8, self.n)
        self.encoder_names = ('a1',)
        self.bias_index = len(self.encoder_names) + 2
        self.readout_end = self.bias_index + 1 + self.context_width + 3
        self.names = self.encoder_names + (
            'phi0', 'phi1', 'bias') + tuple(
            f'Z{i}_weight' for i in range(self.context_width)) + (
            'mean_Z_weight', 'mean_Z2_weight', 'mean_Z3_weight')
        self.names += tuple(coefficient for rank in MODELS[name]
                            for coefficient, _ in rank)
        self.num_parameters = len(self.names)
        self.parameter_indices = {key: i for i, key in enumerate(self.names)}

        self.layers = []
        start = self.readout_end
        for rank, terms in enumerate(MODELS[name], start=2):
            wires = tuple(combinations(range(self.n), rank))
            self.layers.append((wires, slice(start, start + len(terms))))
            start += len(terms)
        self.features = {}
        for sample in samples:
            descriptors = (many_body_features(sample, constants, name)
                           if name in MANY_BODY_MODELS else {})
            self.features[sample.sample_id] = tuple(
                np.stack([descriptors[d] for _, d in terms]) for terms in MODELS[name])

        def circuit(p, a1, phi, *many_body):
            for i in range(self.n):
                angle = 0.5 * math.pi * qml.math.tanh(a1[0] * p[i])
                qml.RY(angle, wires=i)
            for (wires, _), x, coefficients in zip(
                    self.layers, many_body[::2], many_body[1::2]):
                argument = coefficients[0] * x[0]
                for i in range(1, len(coefficients)):
                    argument = argument + coefficients[i] * x[i]
                angles = 0.5 * math.pi * qml.math.tanh(argument)
                for address, angle in zip(wires, angles):
                    if len(address) == 2:
                        qml.IsingZZ(angle, wires=address)
                    else:
                        qml.MultiRZ(angle, wires=address)
            hea(self.n, phi)
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(self.n))

        self.circuit = circuit
        device = qml.device('default.qubit', wires=self.n, shots=None, seed=0)
        self.qnode = qml.QNode(circuit, device, interface='torch', diff_method='backprop')

    def quantum_arguments(self, sample, theta):
        c = self.constants
        if self.name in MANY_BODY_MODELS:
            p = torch.tensor(standardized_density_diagonal(sample, c),
                             dtype=theta.dtype, device=theta.device)
        else:
            p = torch.tensor(sample.pii, dtype=theta.dtype, device=theta.device)
            p = (p - float(c.Pii_mean)) / float(c.Pii_std)
        arguments = [p, theta[:1], theta[1:self.bias_index]]
        for (_, indices), x in zip(self.layers, self.features[sample.sample_id]):
            arguments.extend((torch.tensor(x, dtype=theta.dtype, device=theta.device),
                              theta[indices]))
        return tuple(arguments)

    def quantum_features(self, sample, theta):
        arguments = self.quantum_arguments(sample, theta)
        return torch.stack(self.qnode(*arguments)).reshape(-1)

    def raw_readout_features(self, sample, theta):
        return readout_features(self.quantum_features(sample, theta), self.context_width)

    def predict_batch(self, samples, theta):
        features = [self.raw_readout_features(s, theta) for s in samples]
        weights = theta[self.bias_index:self.readout_end]
        return weights[0] + torch.stack(features, dim=0) @ weights[1:]

    def __call__(self, sample, theta):
        features = self.raw_readout_features(sample, theta)
        weights = theta[self.bias_index:self.readout_end]
        return weights[0] + features @ weights[1:]
