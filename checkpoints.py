"""Load compatible production checkpoints without changing their saved provenance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import DEFAULT_INPUT, constants_from_saved, load_data
from .preprocessing import MANY_BODY_MODELS, HIERARCHY_LAYERS, model_definition
from .models import MODELS, Model
from .registry import MOLECULES
from .train import IMPLEMENTATION, RESULTS_ROOT, create_output_directory, error_metrics, predict, write_csv, write_json

NATIVE_GAMMA_V3 = ('PySCF RMP2.make_rdm2, unrelaxed, frozen=0; Gamma_MP2_AO_pyscf; '
                   'four-axis S_half Lowdin transformation; spin-summed interleaved (bra1,ket1,bra2,ket2)')


def select_checkpoints(source, molecule, name, seeds=None):
    """Resolve only explicit run layouts, never search unrelated nested runs."""
    source = Path(source).expanduser().resolve()
    bases = {source / molecule / name, source / name / molecule / name}
    if source.name == molecule:
        bases.add(source / name)
    if source.name == name and source.parent.name == molecule:
        bases.add(source)
    candidates = [source] if source.name.startswith('seed_') else [
        p for base in sorted(bases) for p in sorted(base.glob('seed_*')) if p.is_dir()]
    indexed = {}
    for p in candidates:
        if p.parent.name != name or p.parent.parent.name != molecule:
            continue
        try:
            seed = int(p.name.removeprefix('seed_'))
        except ValueError:
            raise ValueError(f'Invalid seed directory: {p}') from None
        if seed in indexed:
            raise ValueError(f'Ambiguous {molecule}/{name}/seed_{seed:02d}: {indexed[seed]} and {p}')
        indexed[seed] = p
    requested = sorted(indexed) if seeds is None else list(seeds)
    if len(requested) != len(set(requested)) or any(s < 0 for s in requested):
        raise ValueError('Select unique nonnegative seed IDs')
    missing = sorted(set(requested) - indexed.keys())
    if missing:
        raise FileNotFoundError(f'{source}: missing {molecule}/{name} seeds {missing}; available seeds {sorted(indexed)}')
    return [indexed[s] for s in requested]


def validate_definition(record, name, version, *, csv_row=False):
    for key, value in model_definition(name).items():
        if csv_row and key == 'one_body_parameter_names':
            continue
        if key in ('pipeline_version', 'parameter_schema', 'model_parameter_schema'):
            value = value.replace('paper_source_v5', version)
        if key == 'gamma_provenance' and version == 'paper_source_v3':
            value = NATIVE_GAMMA_V3
        actual = record.get(key)
        matches = str(actual) == str(value) if csv_row else actual == value
        if not matches:
            raise ValueError(f'Incompatible {name} {key}: {actual!r}; expected {value!r}')


def read_checkpoint(directory):
    """Validate saved semantics, layout and completion; never load optimizer state."""
    directory = Path(directory).resolve()
    identity, metadata, saved = [json.loads((directory / f).read_text())
                                 for f in ('result.json', 'parameters.json', 'preprocessing.json')]
    molecule, name, seed = identity['molecule'], identity['model'], identity['seed']
    if molecule not in MOLECULES or name not in MODELS or type(seed) is not int or seed < 0:
        raise ValueError('Unsupported checkpoint molecule, model or seed')
    if (directory.name != f'seed_{seed:02d}' or directory.parent.name != name
            or directory.parent.parent.name != molecule or metadata.get('model') != name
            or any(k in metadata and metadata[k] != identity[k] for k in ('molecule', 'seed'))):
        raise ValueError('Checkpoint molecule/model/seed disagrees with path or metadata')
    implementation = identity.get('implementation')
    if implementation not in (IMPLEMENTATION, 'paper_source_v3_v1') or metadata.get('implementation') != implementation:
        raise ValueError('Unsupported or conflicting checkpoint implementation')
    version = implementation.removesuffix('_v1')
    for record in (identity, metadata):
        validate_definition(record, name, version)
    adapted = dict(saved)
    if name in MANY_BODY_MODELS:
        definition = model_definition(name)
        for key in ('pipeline_version', 'parameter_schema'):
            expected = definition[key].replace('paper_source_v5', version)
            if saved.get(key) != expected:
                raise ValueError(f'Incompatible saved preprocessing {key}')
            adapted[key] = definition[key]
    validated = constants_from_saved(adapted, name)
    constants = type(validated)(**saved)  # Keep the original schema labels and numeric values.
    if name == 'MB1' and (not np.isfinite([constants.Pii_mean, constants.Pii_std]).all()
                         or constants.Pii_std <= 0):
        raise ValueError('Invalid MB1 saved normalization')
    if name == 'MB2_gamm0':
        for rank in (1, 2):
            mean, std, denominator = [getattr(constants, f'{key}_{rank}')
                                      for key in ('mu', 'sigma', 'denominator')]
            if (not np.isfinite([mean, std]).all() or std < 0
                    or denominator != (1.0 if std < 1e-12 else std)):
                raise ValueError('Invalid MB2_gamm0 saved normalization or denominator')
    if (constants.molecule != molecule or constants.n_active != len(MOLECULES[molecule][1])
            or list(constants.training_sample_ids) != [f'{molecule}/{i:03d}' for i in range(1, 31, 2)]):
        raise ValueError('Saved preprocessing molecule, wire count or training population disagrees')
    model = Model(constants, name, ())
    if (metadata['names'] != list(model.names) or metadata['count'] != model.num_parameters
            or metadata['encoder_names'] != list(model.encoder_names)
            or (name in MANY_BODY_MODELS and (metadata.get('parameter_indices') != model.parameter_indices
                                             or metadata.get('readout_bias_name') != 'bias'))):
        raise ValueError('Saved parameter layout does not match the model')
    theta = np.load(directory / 'best_theta.npy', allow_pickle=False)
    if theta.shape != (model.num_parameters,) or not np.isfinite(theta).all():
        raise ValueError('Invalid best checkpoint parameter dimensions or values')
    with (directory / 'training_history.csv').open(newline='') as f:
        history = list(csv.DictReader(f))
    mae = np.array([float(r['train_mae_ha']) for r in history])
    if (not len(history) or not np.isfinite(mae).all()
            or [int(r['epoch']) for r in history] != list(range(1, len(history) + 1))
            or identity.get('update_count') != 2 * len(history)
            or any(r['molecule'] != molecule or r['model'] != name or int(r['seed']) != seed for r in history)
            or identity['best_epoch'] != int(np.argmin(mae)) + 1
            or not np.isclose(identity['best_train_mae_ha'], mae.min(), atol=1e-12, rtol=0)):
        raise ValueError('Incomplete history or checkpoint is not the training-MAE minimum')
    return constants, theta, identity


def saved_tables(directory, identity):
    tables = []
    for filename in ('predictions.csv', 'metrics.csv'):
        with (Path(directory) / filename).open(newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if any(str(row[k]) != str(identity[k]) for k in ('molecule', 'model', 'seed')):
                raise ValueError(f'{filename}: conflicting molecule/model/seed')
            validate_definition(row, identity['model'], identity['implementation'].removesuffix('_v1'), csv_row=True)
        tables.append(rows)
    predictions, metrics = tables
    expected = {(split, f'{i:03d}') for split, start in [('train', 1), ('test', 2)] for i in range(start, 31, 2)}
    if len(predictions) != 30 or {(r['split'], r['geometry_id']) for r in predictions} != expected:
        raise ValueError('Saved prediction membership is incomplete or duplicated')
    if len(metrics) != 2 or {r['split'] for r in metrics} != {'train', 'test'}:
        raise ValueError('Saved metric membership is incomplete or duplicated')
    for metric in metrics:
        rows = [r for r in predictions if r['split'] == metric['split']]
        y, p = [np.array([float(r[k]) for r in rows]) for k in ('target_Ha', 'prediction_Ha')]
        if not np.isfinite([y, p]).all() or int(metric['count']) != 15 or int(metric['best_epoch']) != identity['best_epoch']:
            raise ValueError('Invalid saved predictions or metric checkpoint identity')
        if metric['split'] == 'train' and not np.isclose(float(metric['mae_hartree']),
                identity['best_train_mae_ha'], atol=1e-12, rtol=0):
            raise ValueError('Saved training metric differs from the best checkpoint')
        if not np.allclose([float(r['error_Ha']) for r in rows], p-y, atol=1e-12, rtol=0):
            raise ValueError('Saved signed errors disagree with predictions')
        for key, value in error_metrics(y, p).items():
            if not np.isclose(float(metric[key]), value, atol=1e-10 if key.endswith('_mHa') else 1e-12, rtol=0):
                raise ValueError(f'Saved {metric["split"]} metric disagrees with predictions: {key}')
    return predictions, {r['split']: r for r in metrics}


def validate_statistics(training, c, name):
    """Check saved normalization through training residual moments; do not fit constants."""
    diagonal = np.concatenate([s.pii for s in training])
    if name == 'MB1':
        pools = [(diagonal, c.Pii_mean, c.Pii_std)]
    elif name in ('MB2_gamm0', 'MB2_gamma'):
        gamma = name == 'MB2_gamma'
        pools = [(diagonal, c.mu_P if gamma else c.mu_1, c.std_P if gamma else c.sigma_1),
                 (np.concatenate([s.descriptors['Gamma_2' if gamma else 'Gamma0_2'] for s in training]),
                  c.mu_Gamma if gamma else c.mu_2, c.std_Gamma if gamma else c.sigma_2)]
    else:
        off = ~np.eye(c.n_active, dtype=bool)
        pools = [(diagonal, c.mu_diag, c.std_diag),
                 (np.concatenate([s.pij[off] for s in training]), c.mu_off, c.std_off)]
        if name in HIERARCHY_LAYERS:
            pools += [(np.concatenate([s.descriptors[f'Gamma0_{rank}'] for s in training]), stat['mean'], stat['std'])
                      for rank, stat in c.gamma_statistics.items()]
    for values, mean, std in pools:
        if (not np.isfinite([mean, std]).all() or std < 0
                or abs(np.mean(values - mean)) > 1e-10
                or abs(np.mean((values - mean)**2) - std**2) > 1e-10):
            raise ValueError('Saved training normalization disagrees with selected native descriptors')


def load_checkpoint(directory, input_dir=None, *, samples=None):
    directory = Path(directory).resolve()
    constants, theta, identity = read_checkpoint(directory)
    training, testing = load_data(input_dir or DEFAULT_INPUT, identity['molecule'], include_f=False) if samples is None else samples
    for split, batch, start in [('train', training, 1), ('test', testing, 2)]:
        if [s.sample_id for s in batch] != [f'{identity["molecule"]}/{i:03d}' for i in range(start, 31, 2)]:
            raise ValueError(f'Input {split} membership disagrees with saved population')
    for sample in training + testing:
        with np.load(sample.source_file, allow_pickle=False) as raw:
            if (str(raw['molecule']) != identity['molecule'] or int(raw['point_index']) != sample.geometry_index
                    or str(raw['geometry_id']) != f'{sample.geometry_index:03d}'):
                raise ValueError('Descriptor molecule/geometry identity mismatch')
            if identity['model'] == 'MB2_gamma':
                from .descriptor import RDM_SOURCE, RDM_CONVENTION
                if (str(raw['rdm_source']) != RDM_SOURCE or str(raw['rdm_convention']) != RDM_CONVENTION
                        or int(raw['mp2_frozen_core']) != 0 or int(raw['rdm_frozen_core']) != 0):
                    raise ValueError('MB2_gamma requires native unrelaxed MP2 descriptors, frozen=0')
        if len(sample.pii) != constants.n_active or any(
                list(getattr(sample, key)) != list(getattr(training[0], key))
                for key in ('active_ao_indices', 'source_selected_ao_indices', 'active_ao_labels')):
            raise ValueError('Input AO ordering or wire count mismatch')
    populations = [directory.parents[1] / 'population.json',
                   *sorted((directory.parent / 'source_metadata').rglob('population.json'))]
    for path in populations:
        if path.is_file():
            population = json.loads(path.read_text())
            if any(population[k] != list(getattr(training[0], k)) for k in ('active_ao_indices', 'source_selected_ao_indices', 'active_ao_labels')):
                raise ValueError(f'Saved AO order differs from descriptors: {path}')
            if population['train'] != [s.sample_id for s in training] or population['test'] != [s.sample_id for s in testing]:
                raise ValueError(f'Saved population differs from descriptors: {path}')
    validate_statistics(training, constants, identity['model'])
    model = Model(constants, identity['model'], training + testing)
    model.provenance = dict(checkpoint=str(directory / 'best_theta.npy'), implementation=identity['implementation'],
                            preprocessing=str(directory / 'preprocessing.json'),
                            preprocessing_schema=getattr(constants, 'parameter_schema', None),
                            sha256={f: hashlib.sha256((directory / f).read_bytes()).hexdigest()
                                    for f in ('best_theta.npy', 'preprocessing.json', 'parameters.json', 'result.json')})
    return model, torch.tensor(theta, dtype=torch.float64), training, testing, identity


def replay(directory, input_dir=None, tolerance=1e-8):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Replay tolerance must be positive and finite')
    model, theta, training, testing, identity = load_checkpoint(directory, input_dir)
    saved, saved_metrics = saved_tables(directory, identity)
    indexed = {(r['split'], r['geometry_id']): r for r in saved}
    rows, metrics, differences = [], {}, []
    for split, samples in (('train', training), ('test', testing)):
        values = predict(model, samples, theta)
        metrics[split] = error_metrics([s.target_energy for s in samples], values)
        for key, value in metrics[split].items():
            if not np.isclose(value, float(saved_metrics[split][key]), atol=tolerance * (1000 if key.endswith('_mHa') else 1), rtol=0):
                raise ValueError(f'Saved {split} metric cannot be reproduced: {key}')
        for sample, value in zip(samples, values):
            old = indexed[split, f'{sample.geometry_index:03d}']
            if (abs(float(old['target_Ha']) - sample.target_energy) > tolerance
                    or abs(float(old['q_A']) - sample.scan_coordinate) > 1e-12
                    or abs(float(old['bond_length_A']) - sample.bond_length) > 1e-12):
                raise ValueError('Input target or geometry differs from saved predictions')
            drift = abs(float(value) - float(old['prediction_Ha']))
            differences.append(drift)
            rows.append(dict(molecule=sample.molecule, model=model.name, seed=identity['seed'], split=split,
                             geometry_id=f'{sample.geometry_index:03d}', prediction_Ha=float(value),
                             saved_prediction_Ha=float(old['prediction_Ha']), absolute_drift_Ha=drift))
    maximum = max(differences)
    if not np.isfinite(maximum) or maximum > tolerance:
        raise ValueError(f'Checkpoint replay drift {maximum} exceeds {tolerance}')
    return rows, dict(molecule=identity['molecule'], model=model.name, seed=identity['seed'], count=len(rows), max_absolute_drift_Ha=maximum,
                      tolerance_Ha=tolerance, metrics=metrics, provenance=model.provenance, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, help='Optional new report directory inside v5 results')
    parser.add_argument('--tolerance', type=float, default=1e-8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        rows, report = replay(args.checkpoint, args.input_dir, args.tolerance)
        if args.output_dir is not None:
            output = create_output_directory(args.output_dir, parent=RESULTS_ROOT / 'replays')
            write_csv(output / 'predictions.csv', rows)
            write_json(output / 'replay.json', report)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
