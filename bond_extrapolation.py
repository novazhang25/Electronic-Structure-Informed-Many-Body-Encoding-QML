"""Bond-length extrapolation with frozen v5 checkpoints."""
import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
import torch

from . import descriptor, initialdata
from .checkpoints import load_checkpoint, saved_tables, select_checkpoints
from .data import DEFAULT_INPUT, load_data, load_sample
from .models import DEFAULT_MODELS, MODELS, Model
from .registry import MOLECULE_NAMES
from .train import RESULTS_ROOT, create_output_directory, predict, write_json

REPLAY_ATOL = 1e-8
GEOMETRY_ATOL = 5e-10


def extrapolation_grid(delta, interval, points_per_side):
    lower, upper = np.asarray(interval, dtype=float)
    outer = np.sqrt(5 / 3) * delta
    if (not np.isfinite([delta, lower, upper]).all() or delta <= 0
            or not -outer < lower < upper < outer or points_per_side < 1):
        raise ValueError('The saved scan must lie inside the n=2 window; positive point count required')
    fraction = np.arange(1, points_per_side + 1) / points_per_side
    # Exclude the original endpoints and include the two outer n=2 endpoints.
    return np.concatenate((lower + fraction * (-outer - lower),
                           upper + fraction * (outer - upper)))


def source_scan(training, testing, points_per_side, *, native_gamma):
    samples = sorted(training + testing, key=lambda s: s.geometry_index)
    molecule = samples[0].molecule
    if [s.geometry_index for s in samples] != list(range(1, 31)):
        raise ValueError(f'{molecule}: expected the complete original 30-geometry scan')
    originals, folders = [], set()
    ncore, nactive, nelectrons = initialdata.ACTIVE_SPACES[molecule]
    for sample in samples:
        with np.load(sample.source_file, allow_pickle=False) as raw:
            state = Path(str(raw['input_state_path'])).expanduser()
            if not state.is_absolute() or not state.is_file():
                raise ValueError(f'{sample.source_file}: saved input_state_path is unavailable: {state}')
            folder = state.parent
            if folder.name != f'{sample.geometry_index:03d}' or folder.parent.name != 'scan':
                raise ValueError(f'{state}: expected the saved scan/<ID>/electronic_state.npz layout')
            folders.add(folder.parents[1].resolve())
            point = json.loads((folder / 'point.json').read_text())
            expected = dict(molecule=molecule, point_index=sample.geometry_index,
                            basis='sto-3g', charge=0, spin=0, status='PASS',
                            n_frozen_orbitals=ncore, n_active_orbitals=nactive,
                            n_active_electrons=nelectrons, fci_convention='frozen-core')
            if any(point.get(k) != v for k, v in expected.items()):
                raise ValueError(f'{folder}: source electronic convention differs from v5')
            expected = dict(units='Angstrom', energy_units='Hartree', basis='sto-3g', charge=0,
                            spin=0, input_fci_frozen_core=ncore, input_fci_convention='frozen-core',
                            mp2_frozen_core=0, rdm_frozen_core=0,
                            rdm_convention=descriptor.RDM_CONVENTION)
            if native_gamma:
                expected['rdm_source'] = descriptor.RDM_SOURCE
            if any(k not in raw or raw[k].item() != v for k, v in expected.items()):
                raise ValueError(f'{sample.source_file}: incompatible descriptor convention or Gamma source')
            if (raw['symbols'].tolist() != point['symbols'] or not np.allclose(
                    raw['cartesian_A'], point['cartesian_A'], atol=1e-12, rtol=0)):
                raise ValueError(f'{folder}: source geometry differs from the descriptor geometry')
            for key, value in [('q_A', sample.scan_coordinate), ('bond_length_A', sample.bond_length),
                               ('E_RHF_Ha', sample.rhf_total_energy), ('E_FCI_Ha', sample.fci_total_energy),
                               ('E_corr_Ha', sample.target_energy)]:
                if not np.isclose(point[key], value, atol=1e-10, rtol=0):
                    raise ValueError(f'{folder}: source {key} differs from the descriptors')
        originals.append(point)
    if len(folders) != 1:
        raise ValueError(f'{molecule}: source geometries come from different scans')
    folder = folders.pop()
    document = json.loads((folder / 'result.json').read_text())
    summary, harmonic = document['summary'], document['harmonic']
    units = harmonic['input']['geometry_units']
    if (summary['molecule'] != molecule or summary['n'] != 1 or units != 'Angstrom'
            or document['status'] != 'PASS'):
        raise ValueError(f'{folder}: expected a completed n=1 scan in Angstrom')
    path = initialdata.StretchPath(molecule, harmonic['symbols'], harmonic['equilibrium_cartesian_A'],
                                   harmonic['isotope_average_masses_amu'])
    equilibrium = json.loads((folder / 'equilibrium_energies/point.json').read_text())
    if (equilibrium['symbols'] != path.symbols or not np.allclose(
            equilibrium['cartesian_A'], path.ref, atol=1e-12, rtol=0)
            or not np.isclose(path.s_eq, summary['r_e_A'], atol=1e-12, rtol=0)):
        raise ValueError(f'{folder}: equilibrium geometry and coordinate disagree')
    delta = float(summary['Delta_q_n1_A'])
    expected_delta = np.sqrt(3 * initialdata.HBAR * summary['omega_rad_per_s']
                             / summary['k_N_per_m']) / 1e-10
    if not np.isclose(delta, expected_delta, atol=1e-12, rtol=1e-8):
        raise ValueError(f'{folder}: Delta_1 is not the saved harmonic n=1 half-width')
    # Use explicit saved endpoints; never infer symmetry from observed extrema.
    lengths = initialdata.scan_grid(summary, 30)
    lower, upper = np.asarray([summary['n1_min_A'], summary['n1_max_A']]) - path.s_eq
    saved_scan = {p['point_index']: p for p in document['scan']}
    if len(document['scan']) != 30 or set(saved_scan) != set(range(1, 31)):
        raise ValueError(f'{folder}: incomplete or duplicate scan metadata')
    maximum = 0.0
    for point, length in zip(originals, lengths):
        q = length - path.s_eq
        coords = path(q)
        saved = saved_scan[point['point_index']]
        error = float(np.max(np.abs(coords - point['cartesian_A'])))
        maximum = max(maximum, error)
        if (point['symbols'] != path.symbols or error > GEOMETRY_ATOL
                or not np.isclose(point['q_A'], q, atol=1e-12, rtol=0)
                or not np.isclose(point['bond_length_A'], length, atol=1e-12, rtol=0)
                or not np.isclose(saved['q_A'], q, atol=1e-12, rtol=0)
                or not np.allclose(saved['cartesian_A'], coords, atol=GEOMETRY_ATOL, rtol=0)):
            raise ValueError(f'{folder}/{point["point_index"]:03d}: source geometry/scan differs from v5; '
                             'no rotation or scan replacement is applied')
    grid = extrapolation_grid(delta, (lower, upper), points_per_side)
    source_q = np.array([p['q_A'] for p in originals])
    if (not np.all((grid < lower) | (grid > upper)) or len(np.unique(grid)) != len(grid)
            or np.any(np.isclose(grid[:, None], source_q, atol=1e-12, rtol=0))):
        raise ValueError(f'{molecule}: extrapolation points overlap the original scan')
    records = []
    for index, q in enumerate(grid, 1):
        side = 'compressed' if index <= points_per_side else 'stretched'
        coords = path(q)
        geometry_check = initialdata.validate_scan_geometry(path, q, coords)
        if not geometry_check['passed']:
            raise ValueError(f'{molecule}/{side}/{index}: fixed internal coordinates changed')
        if any(np.allclose(coords, p['cartesian_A'], atol=1e-12, rtol=0) for p in originals):
            raise ValueError(f'{molecule}/{side}/{index}: duplicate source geometry')
        records.append(dict(molecule=molecule, geometry_id=f'{side}_{index:03d}', point_index=index,
                            symbols=path.symbols, cartesian_A=coords.tolist(), q_A=float(q),
                            bond_length_A=float(path.s_eq + q), side=side))
    scan = dict(source=str(folder / 'result.json'), coordinate=summary['coordinate'], units=units,
                r_e_A=path.s_eq, Delta_1_A=delta, Delta_2_A=float(np.sqrt(5 / 3) * delta),
                original_interval_A=[float(lengths[0]), float(lengths[-1])],
                original_q_interval_A=[float(lower), float(upper)],
                training_interval_A=[min(s.bond_length for s in training), max(s.bond_length for s in training)],
                original_points=30, endpoint_convention='30 equally spaced points, both saved endpoints included',
                extension_convention='j=1..N: each saved endpoint + (j/N)*(outer n=2 endpoint - saved endpoint)',
                max_source_geometry_error_A=maximum, points_per_side=points_per_side,
                orientation='COM translation only' if molecule == 'NH3' else 'mass-weighted alignment to equilibrium')
    return scan, records


def frozen_predict(model, samples, theta):
    unlabelled = tuple(replace(s, target_energy=np.nan, rhf_total_energy=np.nan,
                              fci_total_energy=np.nan) for s in samples)
    values = predict(model, unlabelled, theta.detach())
    if values.shape != (len(samples),) or not np.isfinite(values).all():
        raise ValueError('Nonfinite or incomplete frozen predictions')
    return values


def prediction_rows(samples, values, name, seed, epoch, sides):
    return [dict(molecule=s.molecule, model=name, seed=seed, best_epoch=epoch,
                 geometry_id=s.sample_id.split('/')[-1], q_A=s.scan_coordinate,
                 bond_length_A=s.bond_length, coordinate_units='Angstrom', side=side,
                 prediction_Ha=float(value), target_Ha=s.target_energy,
                 signed_error_Ha=float(value - s.target_energy), absolute_error_Ha=float(abs(value - s.target_energy)))
            for s, value, side in zip(samples, values, sides)]


def prepare_sources(args):
    directories = [(m, name, directory) for m in args.molecules for name in args.models
                   for directory in select_checkpoints(args.source_run, m, name, args.seeds)]
    input_dir = args.input_dir.expanduser().resolve()
    source_samples, scans, geometries = {}, {}, {}
    for molecule in args.molecules:
        source_samples[molecule] = load_data(input_dir, molecule, include_f=False)
        scans[molecule], geometries[molecule] = source_scan(*source_samples[molecule], args.points_per_side,
                                                          native_gamma='MB2_gamma' in args.models)
    contexts, replays = [], []
    for molecule, name, directory in directories:
        model, theta, training, testing, identity = load_checkpoint(directory, samples=source_samples[molecule])
        seed = identity['seed']
        predictions, _ = saved_tables(directory, identity)
        saved = pd.DataFrame([r for r in predictions if r['split'] == 'test'])
        for key in ('target_Ha', 'prediction_Ha', 'q_A', 'bond_length_A'):
            saved[key] = pd.to_numeric(saved[key], errors='raise')
        saved = saved.set_index('geometry_id').loc[[f'{s.geometry_index:03d}' for s in testing]]
        if (not np.allclose(saved.target_Ha, [s.target_energy for s in testing], atol=REPLAY_ATOL, rtol=0)
                or not np.allclose(saved.q_A, [s.scan_coordinate for s in testing], atol=1e-12, rtol=0)
                or not np.allclose(saved.bond_length_A, [s.bond_length for s in testing], atol=1e-12, rtol=0)):
            raise ValueError(f'{directory}: source targets/coordinates disagree with saved predictions')
        values = frozen_predict(model, testing, theta)
        drift = np.abs(values - saved.prediction_Ha.to_numpy(float))
        if not np.isfinite(drift).all() or np.max(drift) > REPLAY_ATOL:
            raise ValueError(f'{directory}: FAIL source replay; maximum drift {np.max(drift):.6g} Ha')
        rows = prediction_rows(testing, values, name, seed, identity['best_epoch'], ['original_test'] * 15)
        for row, old, error in zip(rows, saved.prediction_Ha, drift):
            row.update(saved_prediction_Ha=float(old), absolute_replay_drift_Ha=float(error), replay_status='PASS')
        replays.extend(rows)
        contexts.append(dict(molecule=molecule, model=name, seed=seed, directory=directory,
                             epoch=identity['best_epoch'], theta=theta, constants=model.constants,
                             provenance=model.provenance))
        print(f'Replay PASS {molecule}/{name}/seed_{seed:02d}: max drift {np.max(drift):.3g} Ha', flush=True)
    return contexts, source_samples, scans, geometries, replays, input_dir


def generate_samples(output, records, source):
    samples = []
    for record in records:
        molecule, gid = record['molecule'], record['geometry_id']
        folder = output / 'geometries' / molecule / gid
        print(f'Generating {molecule}/{gid}', flush=True)
        point = initialdata.calculate_point(molecule, record['symbols'], record['cartesian_A'],
                    record['bond_length_A'], record['q_A'], record['point_index'], folder, output)
        if point['status'] != 'PASS':
            raise ValueError(f'{molecule}/{gid}: {point.get("error", "electronic calculation failed")}')
        point.update(geometry_id=gid, side=record['side'],
                     input_state_path=str(folder / 'electronic_state.npz'),
                     input_geometry_path=str(folder / 'geometry.xyz'))
        descriptor.save_record(output / 'descriptors', point, descriptor.calculate_record(point))
        sample = load_sample(output / 'descriptors' / molecule / f'{gid}.npz', molecule,
                             record['point_index'], sample_id=f'{molecule}/{gid}', include_f=False)
        for key in ('active_ao_indices', 'source_selected_ao_indices', 'active_ao_labels'):
            if list(getattr(sample, key)) != list(getattr(source, key)):
                raise ValueError(f'{molecule}/{gid}: selected AO order differs from the source: {key}')
        samples.append(sample)
    return tuple(samples)


def summarize(predictions):
    metrics, summaries = [], []
    for (molecule, model, seed), group in predictions.groupby(['molecule', 'model', 'seed'], sort=False):
        for side in ('original_test', 'compressed', 'stretched', 'combined'):
            subset = group[group.side.isin(['compressed', 'stretched'])] if side == 'combined' else group[group.side == side]
            values = subset.signed_error_Ha.to_numpy(float)
            if not len(values) or not np.isfinite(values).all() or subset.geometry_id.duplicated().any():
                raise ValueError(f'{molecule}/{model}/{seed}/{side}: incomplete or invalid predictions')
            metrics.append(dict(molecule=molecule, model=model, seed=int(seed), side=side,
                                count=len(values), mae_Ha=float(np.mean(np.abs(values))),
                                mae_mHa=float(1000 * np.mean(np.abs(values)))))
    metrics = pd.DataFrame(metrics)
    for (molecule, model, side), group in metrics.groupby(['molecule', 'model', 'side'], sort=False):
        if group['count'].nunique() != 1:
            raise ValueError(f'{molecule}/{model}/{side}: unequal geometry counts across seeds')
        values = group.mae_mHa.to_numpy(float)
        mean, count = float(np.mean(values)), len(values)
        std = float(np.std(values, ddof=1)) if count > 1 else np.nan
        half = float(t.ppf(.975, count - 1) * std / np.sqrt(count)) if count > 1 else np.nan
        summaries.append(dict(molecule=molecule, model=model, side=side, seed_count=count,
                              points_per_seed=int(group['count'].iloc[0]), mean_mae_mHa=mean,
                              std_mae_mHa=std, std_ddof=1, ci95_low_mHa=mean-half, ci95_high_mHa=mean+half))
    return metrics, pd.DataFrame(summaries)


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    predictions = pd.read_csv(output / 'predictions.csv')
    summaries = pd.read_csv(output / 'summary.csv')
    scans = json.loads((output / 'metadata.json').read_text())['scans']
    for (molecule, model), group in predictions.groupby(['molecule', 'model'], sort=False):
        folder = output / 'plots' / molecule / model
        folder.mkdir(parents=True, exist_ok=True)
        curve = group.groupby(['side', 'geometry_id', 'bond_length_A'], as_index=False)[['prediction_Ha', 'target_Ha']].mean()
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        reference = curve.sort_values('bond_length_A')
        ax.plot(reference.bond_length_A, 1000 * reference.target_Ha, 'k-', label='FCI − RHF reference')
        for side, color in [('original_test', 'C0'), ('compressed', 'C1'), ('stretched', 'C2')]:
            rows = curve[curve.side == side].sort_values('bond_length_A')
            ax.plot(rows.bond_length_A, 1000 * rows.prediction_Ha, 'o-', ms=3, color=color,
                    label=f'{side.replace("_", " ")}: mean prediction')
        ax.axvspan(*scans[molecule]['original_interval_A'], color='gray', alpha=.12, label='Original 30-point interval')
        ax.set(xlabel='Bond length (Angstrom)', ylabel='Correlation energy (mHa)', title=f'{molecule} / {model}')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        for suffix in ('png', 'pdf'):
            fig.savefig(folder / f'bond_length_curves.{suffix}', dpi=300)
        plt.close(fig)
        rows = summaries[(summaries.molecule == molecule) & (summaries.model == model)]
        rows = rows.set_index('side').loc[['original_test', 'compressed', 'stretched', 'combined']]
        errors = np.array([rows.mean_mae_mHa - rows.ci95_low_mHa, rows.ci95_high_mHa - rows.mean_mae_mHa])
        fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
        ax.bar(np.arange(4), rows.mean_mae_mHa, color=['C0', 'C1', 'C2', 'C3'],
               yerr=errors if np.isfinite(errors).all() else None, capsize=3)
        ax.set(xticks=np.arange(4), xticklabels=['Original test', 'Compressed', 'Stretched', 'Combined'],
               ylabel='Mean per-seed MAE (mHa)', title=f'{molecule} / {model}: 95% Student-t CI when available')
        ax.grid(axis='y', alpha=.2)
        for suffix in ('png', 'pdf'):
            fig.savefig(folder / f'region_mae.{suffix}', dpi=300)
        plt.close(fig)


def run(args):
    if args.output is not None and (args.output.exists() or not args.output.is_relative_to(RESULTS_ROOT.resolve())):
        raise ValueError('Choose a new output directory inside paper_source_v5/results')
    from pyscf import lib
    lib.num_threads(args.threads)
    torch.set_num_threads(args.threads)
    contexts, sources, scans, geometries, replays, input_dir = prepare_sources(args)
    protected = [*[c['directory'].parent for c in contexts], input_dir.resolve(),
                 *[Path(scan['source']).parent for scan in scans.values()]]
    if args.output is not None and any(args.output.is_relative_to(p) for p in protected):
        raise ValueError('Output must be outside the source run, descriptors and initial geometries')
    output = create_output_directory(args.output, parent=RESULTS_ROOT / 'bond_extrapolation')
    metadata = dict(source_run=str(args.source_run), source_descriptors=str(input_dir),
                    molecules=args.molecules, models=args.models, seeds=args.seeds, scans=scans,
                    descriptor_method=descriptor.RDM_SOURCE, pair_extraction='Gamma_L[I,I,J,J]',
                    target='E_FCI(frozen core) - E_RHF, Hartree', replay_tolerance_Ha=REPLAY_ATOL,
                    uncertainty='Sample SD (ddof=1) and 95% Student-t CI of individual-seed MAEs; unavailable for one seed',
                    sources=[dict(molecule=c['molecule'], model=c['model'], seed=c['seed'], best_epoch=c['epoch'],
                                  checkpoint=str(c['directory'] / 'best_theta.npy'),
                                  preprocessing=str(c['directory'] / 'preprocessing.json'),
                                  provenance=c['provenance']) for c in contexts])
    write_json(output / 'metadata.json', metadata)
    pd.DataFrame(replays).to_csv(output / 'source_replay.csv', index=False)
    rows = [{k: v for k, v in row.items() if k not in ('saved_prediction_Ha', 'absolute_replay_drift_Ha', 'replay_status')}
            for row in replays]
    for molecule in args.molecules:
        records = geometries[molecule]
        write_json(output / f'{molecule}_geometries.json', records)
        samples = generate_samples(output, records, sources[molecule][0][0])
        for context in [c for c in contexts if c['molecule'] == molecule]:
            model = Model(context['constants'], context['model'], samples)
            values = frozen_predict(model, samples, context['theta'])
            rows.extend(prediction_rows(samples, values, context['model'], context['seed'], context['epoch'],
                                        [r['side'] for r in records]))
    predictions = pd.DataFrame(rows)
    metrics, summary = summarize(predictions)
    predictions.to_csv(output / 'predictions.csv', index=False)
    metrics.to_csv(output / 'per_seed_metrics.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    if not args.no_plots:
        plot_results(output)
    print(summary.to_string(index=False), flush=True)
    print(f'Results: {output}', flush=True)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True, help='Explicit production root, molecule/model directory, or seed directory')
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT, help='Existing native descriptor directory')
    parser.add_argument('--molecules', nargs='+', choices=[*MOLECULE_NAMES, 'all'], default=['all'])
    parser.add_argument('--models', nargs='+', choices=[*MODELS, 'all'], default=list(DEFAULT_MODELS),
                        help='Default: the eight MB models; explicit all includes FG')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(64)))
    parser.add_argument('--points-per-side', type=int, default=10)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, help='New directory inside paper_source_v5/results')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    for key, registry in [('molecules', MOLECULE_NAMES), ('models', MODELS)]:
        values = getattr(args, key)
        setattr(args, key, list(registry) if values == ['all'] else values)
    if any('all' in values or len(values) != len(set(values)) for values in (args.molecules, args.models, args.seeds)):
        parser.error('Select unique molecules, models and seeds; use all alone')
    if min(args.seeds) < 0 or min(args.points_per_side, args.threads) < 1:
        parser.error('Use nonnegative seeds and positive point/thread counts')
    args.source_run = args.source_run.expanduser().resolve()
    args.output = (args.output or RESULTS_ROOT / 'bond_extrapolation' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        raise SystemExit(f'Error: {error}') from error


if __name__ == '__main__':
    main()
