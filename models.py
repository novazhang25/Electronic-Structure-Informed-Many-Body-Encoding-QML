"""Many-body density encoders and the geometric fingerprint baseline."""
from itertools import combinations
import math

import numpy as np
import pennylane as qml
import torch


DO = (('a2', 'D'), ('b2', 'O'))
ABC = (('a3', 'A'), ('b3', 'B'), ('c3', 'C'))
G2 = (('a2', 'Gamma0_2'),)
G3 = (('a3', 'Gamma0_3'),)
G4 = (('a4', 'Gamma0_4'),)
Q4 = tuple(zip(('a4', 'b4', 'c4', 'd4', 'e4'),
               ('Q_A', 'Q_B', 'Q_C', 'Q_D', 'Q_E')))

MODELS = {
    'MB1': (),
    'MB1_PRIME': (),
    'MB1_DOUBLE_PRIME': (),
    'MB1C': (),
    'MB1F': (),
    'MB2': (DO,),
    'MB2_prime': ((DO[0],),),
    'MB2_double_prime': ((DO[1],),),
    'MB2_Gamma0': ((('a2', 'Gamma0'),),),
    'MB2_Gamma': ((('a2', 'Gamma'),),),
    'MB3': (DO, ABC),
    'MB3_prime': (DO, (ABC[0],)),
    'MB3_double_prime': (DO, (ABC[1],)),
    'MB3_triple_prime': (DO, (ABC[2],)),
    'MB3_Gamma0': (G2, G3),
    'MB3_Gamma0_prime': (DO, G3),
    'mb4_density_full': (DO, ABC, Q4),
    **{f'mb4_density_{letter}': (DO, ABC, (term,))
       for letter, term in zip('abcde', Q4)},
    'MB4_Gamma0': (G2, G3, G4),
    'MB4_Gamma0_prime': ((('a2', 'D'), ('b2', 'O_square')), G3, G4),
    'FG': (),
}


CONTROL_ENCODER_NAMES = {'MB1_DOUBLE_PRIME': ('a1',), 'MB1C': ('c1',), 'MB1F': ('b1', 'c1')}
DISPLAY_LABELS = {'MB1_DOUBLE_PRIME': 'MB1″', 'MB1C': 'MB1C', 'MB1F': 'MB1F'}


def one_body_encoder(n, p, f, abc):
    a, b, c = abc
    for i in range(n):
        angle = 0.5 * math.pi * qml.math.tanh(a * p[i] + b * f[i] + c)
        qml.RY(angle, wires=i)


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
    def __init__(self, constants, name, samples):
        self.constants = constants
        self.name = name
        self.n = constants.n_active
        self.context_width = max(8, self.n)
        encoder_names = (() if name == 'FG' else
                         CONTROL_ENCODER_NAMES[name] if name in CONTROL_ENCODER_NAMES else
                         ('a1', 'c1') if name == 'MB1_PRIME' else ('a', 'b', 'c'))
        self.bias_index = len(encoder_names) + 2
        self.readout_end = self.bias_index + 1 + self.context_width + 3
        self.names = encoder_names + (
            'phi0', 'phi1', 'bias') + tuple(
            f'Z{i}_weight' for i in range(self.context_width)) + (
            'mean_Z_weight', 'mean_Z2_weight', 'mean_Z3_weight')
        self.names += tuple(coefficient for rank in MODELS[name]
                            for coefficient, _ in rank)
        self.num_parameters = len(self.names)

        self.layers = []
        start = self.readout_end
        for rank, terms in enumerate(MODELS[name], start=2):
            wires = tuple(combinations(range(self.n), rank))
            self.layers.append((wires, slice(start, start + len(terms))))
            start += len(terms)
        self.features = {
            s.sample_id: tuple(np.stack([s.descriptors[d] for _, d in terms])
                               for terms in MODELS[name])
            for s in samples
        }

        def circuit(p, f, abc, phi, *many_body):
            if name == 'FG':
                for i in range(self.n):
                    qml.RY(p[i], wires=i)
            elif name == 'MB1_PRIME':
                a1, c1 = abc
                for i in range(self.n):
                    angle = 0.5 * math.pi * qml.math.tanh(a1 * p[i] + c1)
                    qml.RY(angle, wires=i)
            elif name in CONTROL_ENCODER_NAMES:
                for i in range(self.n):
                    if name == 'MB1_DOUBLE_PRIME':
                        argument = abc[0] * p[i]
                    elif name == 'MB1C':
                        argument = abc[0]
                    else:  # MB1F
                        argument = abc[0] * f[i] + abc[1]
                    qml.RY(0.5 * math.pi * qml.math.tanh(argument), wires=i)
            else:
                one_body_encoder(self.n, p, f, abc)

            for (wires, _), x, coefficients in zip(
                    self.layers, many_body[::2], many_body[1::2]):
                argument = coefficients[0] * x[0]
                for i in range(1, len(coefficients)):
                    argument = argument + coefficients[i] * x[i]
                angles = 0.5 * math.pi * qml.math.tanh(argument)
                # IsingZZ and MultiRZ implement exp(-i gamma Z...Z / 2).
                for address, angle in zip(wires, angles):
                    if len(address) == 2:
                        qml.IsingZZ(angle, wires=address)
                    else:
                        qml.MultiRZ(angle, wires=address)

            hea(self.n, phi)
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(self.n))

        self.circuit = circuit
        self.qnode = None
        if name != 'FG':
            device = qml.device('default.qubit', wires=self.n, shots=None, seed=0)
            self.qnode = qml.QNode(circuit, device, interface='torch', diff_method='backprop')

    def quantum_arguments(self, sample, theta):
        c = self.constants
        if self.name in CONTROL_ENCODER_NAMES:
            k = len(CONTROL_ENCODER_NAMES[self.name])
            p, f = None, None
            if self.name == 'MB1_DOUBLE_PRIME':
                p = torch.tensor(sample.pii, dtype=theta.dtype, device=theta.device)
                p = (p - float(c.Pii_mean)) / float(c.Pii_std)
            elif self.name == 'MB1F':
                f = torch.tensor(sample.fii, dtype=theta.dtype, device=theta.device)
                f = (f - float(c.Fii_mean)) / float(c.Fii_std)
            return p, f, theta[:k], theta[k:self.bias_index]
        if self.name == 'FG':
            if self.qnode is None:
                # FG consumes one global NumPy seed draw at its first prediction.
                device = qml.device('default.qubit', wires=self.n)
                self.qnode = qml.QNode(self.circuit, device, interface='torch', diff_method='backprop')
            p = math.pi / 2.0 * (sample.fingerprint - np.asarray(c.mu)) / np.asarray(c.sigma_safe)
            p[np.asarray(c.constant_mask, dtype=bool)] = 0.0
            p = torch.tensor(np.clip(p, -math.pi, math.pi), dtype=theta.dtype, device=theta.device)
            return p, None, None, theta[:2]
        p = torch.tensor(sample.pii, dtype=theta.dtype, device=theta.device)
        if self.name == 'MB1_PRIME':
            p = (p - float(c.Pii_mean)) / float(c.Pii_std)
            return p, None, theta[:2], theta[2:4]
        f = torch.tensor(sample.fii, dtype=theta.dtype, device=theta.device)
        p = (p - float(c.Pii_mean)) / float(c.Pii_std)
        f = (f - float(c.Fii_mean)) / float(c.Fii_std)
        common = theta[:self.readout_end]
        arguments = [p, f, common[:3], common[3:5]]
        for (_, indices), x in zip(self.layers, self.features[sample.sample_id]):
            arguments.extend((torch.tensor(x, dtype=theta.dtype, device=theta.device), theta[indices]))
        return tuple(arguments)

    def quantum_features(self, sample, theta):
        arguments = self.quantum_arguments(sample, theta)
        return torch.stack(self.qnode(*arguments)).reshape(-1)

    def raw_readout_features(self, sample, theta):
        return readout_features(self.quantum_features(sample, theta), self.context_width)

    def predict_batch(self, samples, theta):
        if self.name == 'FG':
            phi = theta[:self.bias_index]
            bias, weights = theta[self.bias_index], theta[self.bias_index + 1:]
            predictions = []
            for sample in samples:
                angles = self.quantum_arguments(sample, theta)[0]
                z = torch.stack(self.qnode(angles, None, None, phi))
                features = readout_features(z, self.context_width)
                predictions.append(bias + torch.dot(weights, features))
            return torch.stack(predictions)
        features = [self.raw_readout_features(s, theta) for s in samples]
        weights = theta[:self.readout_end][self.bias_index:]
        return weights[0] + torch.stack(features, dim=0) @ weights[1:]

    def __call__(self, sample, theta):
        features = self.raw_readout_features(sample, theta)
        if self.name == 'FG':
            return theta[self.bias_index] + torch.dot(theta[self.bias_index + 1:], features)
        weights = theta[:self.readout_end][self.bias_index:]
        return weights[0] + features @ weights[1:]
