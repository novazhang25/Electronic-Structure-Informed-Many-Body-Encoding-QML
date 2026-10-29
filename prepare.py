"""Validate existing input archives and prepare support features in memory."""
import argparse
from pathlib import Path

import numpy as np

from .data import DEFAULT_INPUT, MOLECULES, fit_constants, load_data
from .train import IMPLEMENTATION, RESULTS_ROOT, create_output_directory, write_json


def inspect_inputs(input_dir, molecules):
    reports = []
    for molecule in molecules:
        training, testing = load_data(input_dir, molecule)
        constants = fit_constants(training)
        if not np.isfinite(constants.Pii_std) or constants.Pii_std <= 0:
            raise ValueError(f'{molecule}: invalid training P diagonal standard deviation')
        for sample in training + testing:
            if not np.allclose(sample.pij, sample.pij.T, rtol=0, atol=1e-12):
                raise ValueError(f'{sample.sample_id}: expected real symmetric RHF density')
            for name, values in sample.descriptors.items():
                if not np.isfinite(values).all():
                    raise ValueError(f'{sample.sample_id}: nonfinite descriptor {name}')
        reports.append(dict(molecule=molecule, training_count=len(training), testing_count=len(testing),
                            wires=constants.n_active, source_selected_ao_indices=training[0].source_selected_ao_indices.tolist(),
                            descriptors=sorted(training[0].descriptors),
                            source_files=[str(s.source_file.resolve()) for s in training + testing]))
    return dict(implementation=IMPLEMENTATION, passed=True, input_dir=str(Path(input_dir).resolve()),
                electronic_structure_recomputed=False, dataset_copied=False, molecules=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--molecules', nargs='+', choices=[*MOLECULES, 'all'], default=['CO'])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    molecules = list(MOLECULES) if args.molecules == ['all'] else args.molecules
    if 'all' in molecules:
        parser.error('Use all alone or select explicit molecule IDs')
    report = inspect_inputs(args.input_dir, molecules)
    try:
        output = create_output_directory(args.output_dir, parent=RESULTS_ROOT / 'input_checks')
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    write_json(output / 'inputs.json', report)
    print(f'PASS: existing inputs prepared for {len(molecules)} molecules; no electronic calculations')
    print(f'Report: {output / "inputs.json"}')

if __name__ == '__main__':
    main()
