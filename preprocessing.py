from dataclasses import asdict, dataclass, field, fields
from math import comb

import numpy as np

from .features import coefficient_density_features

PREPROCESSING_METADATA = dict(pipeline_version='paper_source_v5',
                        parameter_schema='paper_source_v5_preprocessing_v1')

_D2 = (('a2', 'Gamma0_2'),)
_D3 = (('a3', 'Gamma0_3'),)
_D4 = (('a4', 'Gamma0_4'),)
HIERARCHY_LAYERS = {
    'MB3_gamm0': (_D2, _D3),
    'MB4_gamma0': (_D2, _D3, _D4),
}

HIERARCHY_SCHEMA = 'paper_source_v5_density_hierarchy_v1'
CYCLE_MODELS = frozenset({'MB3', 'MB4'})
CYCLE_SCHEMA = 'paper_source_v5_standardized_c_encoders_v1'


@dataclass
class MB2Gamm0Constants:
    molecule: str
    training_sample_ids: tuple[str, ...]
    n_active: int
    mu_1: float = 0.0
    sigma_1: float = 1.0
    denominator_1: float = 1.0
    pipeline_version: str = 'paper_source_v5'
    parameter_schema: str = 'paper_source_v5_preprocessing_v1'
    mu_2: float = 0.0
    sigma_2: float = 1.0
    denominator_2: float = 1.0


@dataclass
class MB2Constants:
    molecule: str
    training_sample_ids: tuple[str, ...]
    n_active: int
    pipeline_version: str = 'paper_source_v5'
    parameter_schema: str = 'paper_source_v5_preprocessing_v1'
    model: str = 'MB2'
    mu_diag: float = 0.0
    std_diag: float = 1.0
    mu_off: float = 0.0
    std_off: float = 1.0
    denominator_diag: float = 1.0
    denominator_off: float = 1.0
    training_diag_entry_count: int = 0
    training_off_entry_count: int = 0
    fit_population: str = 'selected_diagonal_and_all_off_diagonal_entries_separately'
    ddof: int = 0


@dataclass
class HierarchyConstants(MB2Constants):
    model: str = ''
    parameter_schema: str = HIERARCHY_SCHEMA
    gamma_statistics: dict = field(default_factory=dict)
    small_std_threshold: float = 1e-12
    small_std_denominator: float = 1.0


@dataclass
class CycleConstants(MB2Constants):
    model: str = ''
    parameter_schema: str = CYCLE_SCHEMA


@dataclass
class MB2GammaConstants:
    molecule: str
    training_sample_ids: tuple[str, ...]
    n_active: int
    mu_P: float = 0.0
    std_P: float = 1.0
    mu_Gamma: float = 0.0
    std_Gamma: float = 1.0
    denominator_P: float = 1.0
    denominator_Gamma: float = 1.0
    training_diagonal_count: int = 0
    training_pair_count: int = 0
    model: str = 'MB2_gamma'
    pair_descriptor: str = 'Gamma_2'
    ddof: int = 0
    pipeline_version: str = 'paper_source_v5'
    parameter_schema: str = 'paper_source_v5_preprocessing_v1'

CONSTANT_TYPES = {
    'MB2': MB2Constants,
    'MB2_gamm0': MB2Gamm0Constants,
    'MB2_gamma': MB2GammaConstants,
}
CONSTANT_TYPES.update({name: HierarchyConstants for name in HIERARCHY_LAYERS})
CONSTANT_TYPES.update({name: CycleConstants for name in CYCLE_MODELS})
MANY_BODY_MODELS = frozenset(CONSTANT_TYPES)


def constants_from_saved(saved, name):
    kind = CONSTANT_TYPES[name]
    if set(saved) != {field.name for field in fields(kind)}:
        raise ValueError('Unexpected preprocessing schema')
    constants = kind(**saved)
    if (constants.pipeline_version != kind.pipeline_version
            or constants.parameter_schema != kind.parameter_schema):
        raise ValueError('Incompatible model preprocessing schema')
    if name == 'MB2_gamm0':
        return constants

    count, n = len(constants.training_sample_ids), constants.n_active
    if constants.model != name or constants.ddof != 0 or count <= 0 or n < 2:
        raise ValueError(f'Invalid {name} fitting population or model identity')
    if name in CYCLE_MODELS and (
            n < (3 if name == 'MB3' else 4)
            or len(set(constants.training_sample_ids)) != count):
        raise ValueError(f'Invalid {name} fitting population or AO support size')
    if isinstance(constants, MB2Constants):
        if (constants.training_diag_entry_count != count * n
                or constants.training_off_entry_count != count * n * (n - 1)
                or constants.fit_population != 'selected_diagonal_and_all_off_diagonal_entries_separately'):
            raise ValueError(f'Invalid {name} diagonal/off-diagonal fitting populations')
        statistics = [(constants.mu_diag, constants.std_diag, constants.denominator_diag),
                      (constants.mu_off, constants.std_off, constants.denominator_off)]
        if name in HIERARCHY_LAYERS:
            if (len(set(constants.training_sample_ids)) != count
                    or n < len(HIERARCHY_LAYERS[name]) + 1
                    or constants.small_std_threshold != 1e-12
                    or constants.small_std_denominator != 1.0):
                raise ValueError(f'Invalid {name} preprocessing identity or convention')
            orders = range(2, len(HIERARCHY_LAYERS[name]) + 2)
            if set(constants.gamma_statistics) != {str(rank) for rank in orders}:
                raise ValueError(f'Invalid {name} Gamma0 descriptor orders')
            for rank in orders:
                stat = constants.gamma_statistics[str(rank)]
                if (set(stat) != {'mean', 'std', 'denominator', 'training_support_count'}
                        or stat['training_support_count'] != count * comb(n, rank)):
                    raise ValueError(f'Invalid {name} Gamma0_{rank} fitting population')
                statistics.append((stat['mean'], stat['std'], stat['denominator']))
    else:
        if (constants.pair_descriptor != 'Gamma_2'
                or constants.training_diagonal_count != count * n
                or constants.training_pair_count != count * n * (n - 1) // 2):
            raise ValueError('Invalid MB2_gamma fitting population or descriptor identity')
        statistics = [(constants.mu_P, constants.std_P, constants.denominator_P),
                      (constants.mu_Gamma, constants.std_Gamma, constants.denominator_Gamma)]
    for mean, std, denominator in statistics:
        if not np.isfinite(mean) or not np.isfinite(std) or std < 0:
            raise ValueError(f'Invalid {name} statistics')
        if denominator != (1.0 if std < 1e-12 else std):
            raise ValueError(f'Invalid {name} effective denominator')
    return constants


def population_statistics(values):
    mean, std = float(np.mean(values)), float(np.std(values, ddof=0))
    return mean, std, 1.0 if std < 1e-12 else std


def fit_hierarchy_constants(training, name):
    values = asdict(fit_many_body_constants(training, 'MB2'))
    values.update(model=name, parameter_schema=HIERARCHY_SCHEMA)
    statistics = {}
    for rank in range(2, len(HIERARCHY_LAYERS[name]) + 2):
        completed = np.concatenate([sample.descriptors[f'Gamma0_{rank}'] for sample in training])
        mean, std, denominator = population_statistics(completed)
        statistics[str(rank)] = dict(mean=mean, std=std, denominator=denominator,
                                     training_support_count=int(completed.size))
    return HierarchyConstants(**values, gamma_statistics=statistics)


def fit_many_body_constants(training, name):
    if name not in MANY_BODY_MODELS:
        raise ValueError(f'Unknown paper_source_v5 many-body model: {name}')
    if name in HIERARCHY_LAYERS:
        return fit_hierarchy_constants(training, name)
    if name in CYCLE_MODELS:
        values = asdict(fit_many_body_constants(training, 'MB2'))
        values.update(model=name, parameter_schema=CYCLE_SCHEMA)
        return CycleConstants(**values)
    n = len(training[0].pii)
    identity = dict(molecule=training[0].molecule, n_active=n,
                    training_sample_ids=tuple(s.sample_id for s in training))
    if name == 'MB2':
        off_mask = ~np.eye(n, dtype=bool)
        diagonal = np.concatenate([np.diag(s.pij) for s in training])
        off_diagonal = np.concatenate([s.pij[off_mask] for s in training])
        mu_diag, std_diag, denominator_diag = population_statistics(diagonal)
        mu_off, std_off, denominator_off = population_statistics(off_diagonal)
        return MB2Constants(
            **identity, mu_diag=mu_diag, std_diag=std_diag, mu_off=mu_off, std_off=std_off,
            denominator_diag=denominator_diag, denominator_off=denominator_off,
            training_diag_entry_count=int(diagonal.size),
            training_off_entry_count=int(off_diagonal.size))

    diagonal = np.concatenate([s.pii for s in training])
    mu_1, sigma_1, denominator_1 = population_statistics(diagonal)
    descriptor = 'Gamma_2' if name == 'MB2_gamma' else 'Gamma0_2'
    pairs = np.concatenate([s.descriptors[descriptor] for s in training])
    mu_2, sigma_2, denominator_2 = population_statistics(pairs)
    if name == 'MB2_gamma':
        return MB2GammaConstants(
            **identity, mu_P=mu_1, std_P=sigma_1, mu_Gamma=mu_2, std_Gamma=sigma_2,
            denominator_P=denominator_1, denominator_Gamma=denominator_2,
            training_diagonal_count=int(diagonal.size), training_pair_count=int(pairs.size))
    return MB2Gamm0Constants(**identity, mu_1=mu_1, sigma_1=sigma_1,
                         denominator_1=denominator_1, mu_2=mu_2, sigma_2=sigma_2,
                         denominator_2=denominator_2)


def standardized_density_diagonal(sample, constants):
    if isinstance(constants, (HierarchyConstants, CycleConstants)):
        return np.diag(standardized_split_density_matrix(sample, constants))
    if isinstance(constants, MB2Constants):
        return (np.diag(np.asarray(sample.pij)) - constants.mu_diag) / constants.std_diag
    if isinstance(constants, MB2GammaConstants):
        return (np.asarray(sample.pii) - constants.mu_P) / constants.std_P
    return (np.asarray(sample.pii) - constants.mu_1) / constants.sigma_1


def standardized_split_density_matrix(sample, constants):
    raw = np.asarray(sample.pij)
    transformed = (raw - constants.mu_off) / constants.denominator_off
    np.fill_diagonal(transformed, (np.diag(raw) - constants.mu_diag) / constants.denominator_diag)
    return transformed


def many_body_features(sample, constants, name):
    if name in CYCLE_MODELS:
        return coefficient_density_features(standardized_split_density_matrix(sample, constants))
    if name in HIERARCHY_LAYERS:
        return {f'Gamma0_{rank}': (sample.descriptors[f'Gamma0_{rank}'] - stat['mean']) / stat['denominator']
                for rank, stat in constants.gamma_statistics.items()}
    if name == 'MB2':
        p = standardized_split_density_matrix(sample, constants)
        i, j = np.triu_indices(constants.n_active, k=1)
        return dict(D=p[i, i] * p[j, j], O=p[i, j] * p[j, i])
    if name == 'MB2_gamm0':
        return {'Gamma0_2': (sample.descriptors['Gamma0_2'] - constants.mu_2)
                / constants.denominator_2}
    return {'Gamma_2': (sample.descriptors['Gamma_2'] - constants.mu_Gamma)
            / constants.denominator_Gamma}


def hierarchy_definition(name):
    layers = HIERARCHY_LAYERS[name]
    coefficients = ['a1'] + [coefficient for terms in layers for coefficient, _ in terms]
    kinds = [f'D{rank}' for rank in range(2, len(layers) + 2)]
    expressions = {
        'D2': 'a2*Z2', 'D3': 'a3*Z3', 'D4': 'a4*Z4',
    }
    definition = dict(
        pipeline_version='paper_source_v5', parameter_schema=HIERARCHY_SCHEMA,
        model_parameter_schema=f'{HIERARCHY_SCHEMA}:{name}', canonical_model=name,
        display_label=name, layer_composition=','.join(kinds),
        one_body_parameter_names=['a1'], one_body_parameter_count=1,
        one_body_definition='(pi/2)*tanh(a1*Qii)',
        encoding_parameter_names=','.join(coefficients),
        encoding_parameter_count=len(coefficients),
        model_diagonal_density_input='training-standardized',
        density_basis='raw selected Lowdin AO P_L; existing molecule AO selection and ordering',
        support_order='unique increasing selected-AO supports in lexicographic combination order',
        preprocessing_fit_population='mu_diag,std_diag: all training P_ii; mu_off,std_off: all training P_ij with i!=j, including both P_ij and P_ji; pooled scalars per molecule; ddof=0',
        preprocessing_definition='Qii=(Pii-mu_diag)/denominator_diag; Qij=(Pij-mu_off)/denominator_off for i!=j; raw P is retained separately',
        effective_denominator_definition='denominator=1 if std<1e-12 else std; applied to diagonal, off-diagonal and completed Gamma0 descriptors at every order',
        gamma_definition='Gamma0_k[I,I]=sum_pi sign(pi)*2**(cycles(pi)-k)*prod_r P[i_r,i_pi(r)]; all permutations and cycles including fixed points; raw P; no factorial or support normalization',
        gamma_preprocessing_definition='Zk=(Gamma0_k(raw P)-mu_k)/denominator_k; fit each completed descriptor over training geometries and unique increasing supports; pooled scalar per order; ddof=0',
        gate_definition='IsingZZ pairs; MultiRZ triples/quadruples; exp(-i*gamma*Z...Z/2); gamma=(pi/2)*tanh(u)',
        parameter_training='independent complete vector; no loaded or frozen inherited coefficients; normal trainable readout bias',
        pair_baseline='MB2_gamm0',
    )
    for key, kind in zip(('pair_definition', 'triple_definition', 'quadruple_definition'), kinds):
        definition[key] = f'(pi/2)*tanh({expressions[kind]})'
    return definition


def cycle_definition(name):
    coefficients = ['a1', 'a2', 'b2', 'a3', 'b3', 'c3']
    if name == 'MB4':
        coefficients.extend(('a4', 'b4', 'c4', 'd4', 'e4'))
    definition = dict(
        pipeline_version='paper_source_v5', parameter_schema=CYCLE_SCHEMA,
        model_parameter_schema=f'{CYCLE_SCHEMA}:{name}', canonical_model=name,
        display_label=name,
        layer_composition='C1,C2,C3' + (',C4' if name == 'MB4' else '') + ',HEA,readout',
        one_body_parameter_names=['a1'], one_body_parameter_count=1,
        encoding_parameter_names=','.join(coefficients),
        encoding_parameter_count=len(coefficients),
        model_diagonal_density_input='training-standardized',
        density_basis='raw selected Lowdin AO P_L; existing molecule AO selection and ordering',
        support_order='unique increasing selected-AO supports in lexicographic combination order',
        preprocessing_fit_population='mu_diag,std_diag: all training P_ii; mu_off,std_off: all training P_ij with i!=j, including both P_ij and P_ji; pooled scalars per molecule; ddof=0',
        preprocessing_definition='Xii=(Pii-mu_diag)/denominator_diag; Xij=(Pij-mu_off)/denominator_off for i!=j; every C1/C2/C3/C4 factor uses this same X; no product or group standardization',
        effective_denominator_definition='denominator=1 if std<1e-12 else std; identical diagonal/off-diagonal transforms in every encoding layer',
        inference_preprocessing='reuse saved training statistics unchanged; never fit on test or inference inputs',
        one_body_definition='(pi/2)*tanh(a1*Xii)',
        pair_definition='(pi/2)*tanh(a2*Xii*Xjj+b2*Xij*Xji)',
        triple_definition='(pi/2)*tanh(a3*Xii*Xjj*Xkk+b3*(Xii*Xjk**2+Xjj*Xik**2+Xkk*Xij**2)+c3*Xij*Xik*Xjk)',
        triple_group_monomial_counts='1,3,1',
        coefficient_convention='one independent coefficient per group shared across all supports of its order; brackets are sums; undirected cycles counted once',
        gate_definition='RY singles; IsingZZ pairs; MultiRZ triples/quadruples; exp(-i*theta*Z...Z/2) for many-body gates; theta=(pi/2)*tanh(complete weighted preactivation)',
        parameter_training='current seeded initialization; independent trainable encoding coefficients; existing HEA and readout; no F input or one-body encoder bias',
    )
    if name == 'MB4':
        definition.update(
            quadruple_definition='(pi/2)*tanh(a4*Xii*Xjj*Xkk*Xll+b4*(Xii*Xjj*Xkl**2+Xii*Xkk*Xjl**2+Xii*Xll*Xjk**2+Xjj*Xkk*Xil**2+Xjj*Xll*Xik**2+Xkk*Xll*Xij**2)+c4*(Xij**2*Xkl**2+Xik**2*Xjl**2+Xil**2*Xjk**2)+d4*(Xii*Xjk*Xjl*Xkl+Xjj*Xik*Xil*Xkl+Xkk*Xij*Xil*Xjl+Xll*Xij*Xik*Xjk)+e4*(Xij*Xjk*Xkl*Xil+Xij*Xjl*Xkl*Xik+Xik*Xjk*Xjl*Xil))',
            quadruple_group_monomial_counts='1,6,3,4,3')
    return definition


def model_definition(name):
    if name == 'FG':
        from .FG import model_definition as fg_definition
        return fg_definition()
    if name in CYCLE_MODELS:
        return cycle_definition(name)
    if name in HIERARCHY_LAYERS:
        return hierarchy_definition(name)
    if name not in MANY_BODY_MODELS:
        return {}
    definition = dict(
        PREPROCESSING_METADATA, canonical_model=name,
        one_body_parameter_names=['a1'], one_body_parameter_count=1,
        model_parameter_schema=f'{PREPROCESSING_METADATA["parameter_schema"]}:{name}',
        model_diagonal_density_input='training-standardized')
    if name == 'MB2_gamma':
        definition.update(
            one_body_definition='(pi/2)*tanh(a1*(P_ii-mu_P)/std_P)',
            encoding_parameter_names='a1,a2', encoding_parameter_count=2,
            preprocessing_fit_population='mu_P,std_P: all selected diagonal entries of training geometries; ddof=0',
            pair_preprocessing_fit_population='mu_Gamma,std_Gamma: complete correlated Gamma_2 values over training geometries and unique i<j pairs; ddof=0',
            preprocessing_definition='P_std_ii=(P_ii-mu_P)/std_P; Gamma_std=(Gamma_2-mu_Gamma)/denominator_Gamma; no further standardization',
            effective_denominator_definition='one-body: std_P directly; pair: denominator_Gamma=1 if std_Gamma<1e-12 else std_Gamma',
            pair_definition='(pi/2)*tanh(a2*Gamma_std_ij); i<j',
            pair_descriptor='Gamma_2',
            gamma_provenance='All molecules: native unrelaxed PySCF RMP2.make_rdm2, frozen=0, Gamma_MP2_AO_pyscf; full four-axis S_half Lowdin transformation; spin-summed interleaved (bra1,ket1,bra2,ket2)',
            gamma_pair_extraction='Gamma_L[I,I,J,J] for mapped full-AO indices I,J of selected unique pairs i<j')
        return definition
    if name == 'MB2':
        definition.update(
            pair_definition='(pi/2)*tanh(a2*P_std_ii*P_std_jj+b2*P_std_ij*P_std_ji); i<j; no further standardization',
            transformed_density_role='P_std is a transformed ML feature matrix, not a physical 1RDM',
            one_body_definition='(pi/2)*tanh(a1*(P_ii-mu_diag)/std_diag)',
            preprocessing_fit_population='mu_diag,std_diag: all training P_ii; mu_off,std_off: all training P_ij with i!=j, including both P_ij and P_ji; pooled scalars per molecule; ddof=0',
            preprocessing_definition='P_std_ii=(P_ii-mu_diag)/denominator_diag; P_std_ij=(P_ij-mu_off)/denominator_off for i!=j',
            effective_denominator_definition='one-body: std_diag directly; pair: denominator_k=1 if std_k<1e-12 else std_k; k=diag,off')
        return definition
    definition.update(
        one_body_definition='(pi/2)*tanh(a1*(P_ii-mu_1)/sigma_1)',
        preprocessing_fit_population='mu_1,sigma_1: all selected diagonal entries of training geometries; ddof=0',
        effective_denominator_definition='one-body: sigma_1 directly; pair: denominator_k=1 if sigma_k<1e-12 else sigma_k',
        preprocessing_definition='G0=P_ii*P_jj-0.5*P_ij*P_ji from raw P; G0_std=(G0-mu_2)/denominator_2',
        pair_definition='(pi/2)*tanh(a2*G0_std_ij); i<j',
        pair_preprocessing_fit_population='mu_2,sigma_2: complete raw-P G0 values over training geometries and unique i<j pairs; ddof=0')
    return definition
