"""Train correlation energies with Adam and select the lowest training MAE."""
import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from .data import DEFAULT_INPUT, MOLECULES, fit_constants, load_data
from .models import MODELS, Model

EPOCHS, BATCH_SIZE, LEARNING_RATE = 500, 8, 0.02
PROJECT = Path(__file__).resolve().parents[1]


def initialize(model, samples, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    if model.name == 'FG':
        parts = [0.05 * torch.randn(shape, dtype=torch.float64, generator=generator)
                 for shape in ((2,), (), (model.num_parameters - 3,))]
        theta = torch.cat([part.reshape(-1) for part in parts])
    else:
        theta = 0.05 * torch.randn(model.num_parameters, dtype=torch.float64,
                                  generator=generator)
    # The random draw at the overwritten bias position is still consumed.
    theta[model.bias_index] = np.mean([float(s.target_energy) for s in samples])
    return theta.requires_grad_()


@torch.no_grad()
def predict(model, samples, theta, batch_size=BATCH_SIZE):
    return np.concatenate([
        model.predict_batch(samples[i:i + batch_size], theta).numpy()
        for i in range(0, len(samples), batch_size)])


def error_metrics(targets, predictions):
    y, p = np.asarray(targets), np.asarray(predictions)
    e = p - y
    values = dict(mae=float(np.mean(np.abs(y - p))),
                  rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                  max_abs_error=float(np.max(np.abs(e))),
                  signed_error_amplitude=float(np.max(e) - np.min(e)))
    return {key + unit: scale * value for key, value in values.items()
            for unit, scale in (('_hartree', 1.0), ('_mHa', 1000.0))}


def train(model, samples, seed, epochs=EPOCHS, batch_size=BATCH_SIZE,
          learning_rate=LEARNING_RATE):
    theta = initialize(model, samples, seed)
    initial = theta.detach().clone()
    i = model.bias_index
    if model.name == 'FG':
        parameters = [theta[:i].detach().requires_grad_(),
                      theta[i].detach().requires_grad_(),
                      theta[i + 1:].detach().requires_grad_()]
    else:
        parameters = [theta]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    # Initialization and shuffling use independent generators with the same seed.
    shuffle = torch.Generator(device='cpu').manual_seed(seed)
    targets = np.asarray([s.target_energy for s in samples])
    history, best_mae, updates = [], float('inf'), 0
    for epoch in range(1, epochs + 1):
        indices = torch.randperm(len(samples), generator=shuffle).tolist()
        for start in range(0, len(indices), batch_size):
            batch = [samples[j] for j in indices[start:start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            theta.grad = None
            y = torch.stack([torch.as_tensor(s.target_energy, dtype=torch.float64)
                             .reshape(()) for s in batch])
            loss = torch.mean((model.predict_batch(batch, theta) - y) ** 2)
            loss.backward()
            if model.name == 'FG':
                for parameter, grad in zip(parameters,
                        (theta.grad[:i], theta.grad[i], theta.grad[i + 1:])):
                    parameter.grad = grad.detach()
            optimizer.step()
            updates += 1
        train_mae = float(np.mean(np.abs(targets - predict(model, samples, theta, batch_size))))
        history.append(dict(epoch=epoch, train_mae_ha=train_mae,
                            train_mae_mha=1000.0 * train_mae))
        if train_mae < best_mae:
            best_mae, best_epoch = train_mae, epoch
            best = theta.detach().clone()
            best_optimizer = copy.deepcopy(optimizer.state_dict())
    final = theta.detach().clone()
    with torch.no_grad():
        theta.copy_(best)
    return dict(theta=theta, initial_theta=initial, final_theta=final,
                best_epoch=best_epoch, best_train_mae_ha=best_mae, history=history,
                best_optimizer=best_optimizer, final_optimizer=optimizer.state_dict(),
                update_count=updates)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def save_run(directory, model, training, testing, result, seed):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'preprocessing.json', asdict(model.constants))
    write_json(directory / 'parameters.json', dict(names=model.names, count=model.num_parameters))
    for name in ('initial_theta', 'final_theta'):
        np.save(directory / f'{name}.npy', result[name].numpy())
    np.save(directory / 'best_theta.npy', result['theta'].detach().numpy())
    for name in ('best_optimizer', 'final_optimizer'):
        torch.save(result[name], directory / f'{name}.pt')
    write_csv(directory / 'training_history.csv', result['history'])
    predictions, metrics = [], []
    identity = dict(molecule=training[0].molecule, model=model.name, seed=seed)
    for split, samples in (('train', training), ('test', testing)):
        values = predict(model, samples, result['theta'])
        targets = [s.target_energy for s in samples]
        metrics.append(dict(identity, split=split, count=len(samples),
                            best_epoch=result['best_epoch'], **error_metrics(targets, values)))
        for sample, value in zip(samples, values):
            predictions.append(dict(identity, split=split, geometry_id=f'{sample.geometry_index:03d}',
                q_A=sample.scan_coordinate, bond_length_A=sample.bond_length,
                target_Ha=sample.target_energy, prediction_Ha=float(value),
                error_Ha=float(value - sample.target_energy)))
    write_csv(directory / 'predictions.csv', predictions)
    write_csv(directory / 'metrics.csv', metrics)
    write_json(directory / 'result.json', dict(identity, best_epoch=result['best_epoch'],
        best_train_mae_ha=result['best_train_mae_ha'], update_count=result['update_count']))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=[*MODELS, 'all'], default=['MB1'])
    parser.add_argument('--molecules', nargs='+', choices=[*MOLECULES, 'all'], default=['CO'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    names = list(MODELS) if args.models == ['all'] else args.models
    molecules = list(MOLECULES) if args.molecules == ['all'] else args.molecules
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output_dir or PROJECT / 'results_paper_v2' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if output.resolve().is_relative_to(Path(__file__).resolve().parent):
        parser.error('Choose an output directory outside paper_source_v2')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'configuration.json', dict(models=names, molecules=molecules,
        seeds=args.seeds, epochs=args.epochs, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        optimizer='Adam', loss='MSE', checkpoint_selection='train', input_dir=str(args.input_dir.resolve())))
    all_metrics = []
    for molecule in molecules:
        training, testing = load_data(args.input_dir, molecule)
        (output / molecule).mkdir()
        write_json(output / molecule / 'population.json', dict(
            train=[s.sample_id for s in training], test=[s.sample_id for s in testing],
            active_ao_indices=training[0].active_ao_indices.tolist(),
            source_selected_ao_indices=training[0].source_selected_ao_indices.tolist(),
            active_ao_labels=training[0].active_ao_labels))
        for name in names:
            model = Model(fit_constants(training, name), name, training + testing)
            for seed in args.seeds:
                print(f'Training {molecule} / {name} / seed {seed}: {args.epochs} epochs', flush=True)
                result = train(model, training, seed, args.epochs)
                rows = save_run(output / molecule / name / f'seed_{seed:02d}',
                                model, training, testing, result, seed)
                all_metrics.extend(rows)
                print(f"Test MAE: {rows[1]['mae_mHa']:.8g} mHa", flush=True)
    write_csv(output / 'metrics.csv', all_metrics)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()
