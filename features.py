from functools import lru_cache
from itertools import combinations, permutations

import numpy as np


def _cycle_type(permutation):
    seen, lengths = set(), []
    for start in range(len(permutation)):
        if start in seen:
            continue
        index, length = start, 0
        while index not in seen:
            seen.add(index)
            length += 1
            index = permutation[index]
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


@lru_cache(maxsize=None)
def cycle_groups(order):
    if order not in (2, 3, 4):
        raise ValueError('Density cycle groups support orders 2, 3, and 4')
    groups = {}
    for permutation in permutations(range(order)):
        groups.setdefault(_cycle_type(permutation), []).append(permutation)
    return {kind: tuple(group) for kind, group in groups.items()}


def density_features(P):
    P = np.asarray(P)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError('P must be a square density matrix')
    if not np.isrealobj(P):
        raise ValueError('The reference pipeline requires a real density matrix')
    P = np.asarray(P, dtype=np.float64)
    if not np.all(np.isfinite(P)):
        raise ValueError('P must contain only finite density entries')
    pairs, triples, quadruples = (
        np.asarray(list(combinations(range(len(P)), order)), dtype=np.int64).reshape(-1, order)
        for order in (2, 3, 4))
    i, j = pairs.T
    values = dict(D=P[i, i] * P[j, j], O=P[i, j] * P[j, i])
    i, j, k = triples.T
    values.update(A=P[i, i] * P[j, j] * P[k, k],
                  B=P[i, i] * P[j, k] ** 2,
                  C=P[i, j] * P[i, k] * P[j, k])
    i, j, k, l = quadruples.T
    values.update(Q_A=P[i, i] * P[j, j] * P[k, k] * P[l, l],
                  Q_B=P[i, i] * P[j, j] * P[k, l] ** 2,
                  Q_C=P[i, j] ** 2 * P[k, l] ** 2,
                  Q_D=P[i, j] * P[i, k] * P[j, k] * P[l, l],
                  Q_E=P[i, j] * P[i, k] * P[j, l] * P[k, l])
    for supports, names in (
        (pairs, {'S11': (1, 1), 'S2': (2,)}),
        (triples, {'S111': (1, 1, 1), 'S21': (2, 1), 'S3': (3,)}),
        (quadruples, {'S1111': (1, 1, 1, 1), 'S211': (2, 1, 1),
                      'S22': (2, 2), 'S31': (3, 1), 'S4': (4,)}),
    ):
        groups = cycle_groups(supports.shape[1])
        for name, kind in names.items():
            value = np.zeros(len(supports), dtype=np.float64)
            for permutation in groups[kind]:
                value += np.prod(P[supports, supports[:, permutation]], axis=1)
            values[name] = value
    values['Gamma0_2'] = values['S11'] - 0.5 * values['S2']
    values['Gamma0_3'] = (values['S111'] - 0.5 * values['S21']
                          + 0.25 * values['S3'])
    values['Gamma0_4'] = (values['S1111'] - 0.5 * values['S211']
                          + 0.25 * values['S22'] + 0.25 * values['S31']
                          - 0.125 * values['S4'])
    return values


def coefficient_density_features(X):
    values = density_features(X)
    # Undirected cycles occur twice in the permutation sums.
    for key in ('S3', 'S31', 'S4'):
        values[key] *= 0.5
    return values
