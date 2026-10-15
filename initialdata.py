"""RHF/frozen-core FCI scans and full-space density descriptors in STO-3G."""
import argparse
from datetime import datetime
from itertools import combinations, permutations
import math
import json
from pathlib import Path

import numpy as np
import pyscf
from scipy.linalg import expm
from scipy.optimize import minimize
from pyscf import gto, lib, mcscf, scf
from pyscf.data import nist
from pyscf.soscf import newton_ah
from pyscf.ci import cisd
from pyscf.fci import direct_spin1
from pyscf.hessian import thermo
from pyscf.mp import mp2

from .data import MOLECULES, density_descriptors

PROJECT = Path(__file__).resolve().parents[1]
ACTIVE_SPACES = {
    #(number of frozen spatial orbitals, active spatial orbitals, active electrons)
    'LiH': (1, 5, 2), 'BeH2': (1, 6, 4), 'H2O': (1, 6, 8),
    'NH3': (1, 7, 8), 'N2': (2, 8, 10), 'CO': (2, 8, 10),
    'HF': (1, 5, 8), 'H2S': (5, 6, 8), 'H2O2': (2, 10, 14),
}
RDM_CONVENTION = 'Gamma[p,q,r,s]=sum_spin <p^dagger r^dagger s q>; P[p,q]=sum_spin <q^dagger p>'
RDM_SOURCE = 'normalized truncated MP2 wavefunction, expectation values through second order'


DIATOMICS = ('LiH', 'N2', 'CO', 'HF')
STARS = {'BeH2': 'Be', 'H2O': 'O', 'NH3': 'N', 'H2S': 'S'}
BOHR_A = nist.BOHR


def starting_geometries(name):
    if name in DIATOMICS:
        symbols = {'LiH': ['Li', 'H'], 'N2': ['N', 'N'],
                   'CO': ['C', 'O'], 'HF': ['H', 'F']}[name]
        length = {'LiH': 1.8, 'N2': 1.2, 'CO': 1.25, 'HF': 1.05}[name]
        return [(symbols, np.array([[0., 0., 0.], [0., 0., length]]))]
    if name == 'BeH2':
        return [(['Be', 'H', 'H'], np.array([[0., 0., 0.], [0., 0., -1.4], [0., 0., 1.4]]))]
    if name in ('H2O', 'H2S'):
        x, z = (.8, .65) if name == 'H2O' else (1., .9)
        return [(['O' if name == 'H2O' else 'S', 'H', 'H'],
                 np.array([[0., 0., 0.], [x, 0., z], [-x, 0., z]]))]
    if name == 'NH3':
        coords = [[0., 0., 0.]] + [[math.cos(p), math.sin(p), .4]
                                   for p in (0., 2 * math.pi / 3, 4 * math.pi / 3)]
        return [(['N', 'H', 'H', 'H'], np.array(coords))]
    return [(['O', 'O', 'H', 'H'], np.array([[0., 0., -.75], [0., 0., .75],
             [1., 0., -1.], [math.cos(p), math.sin(p), 1.]]))
            for p in np.deg2rad([60., 120., 160.])]


def make_molecule(symbols, coords, basis='sto-3g'):
    return gto.M(atom=list(zip(symbols, np.asarray(coords).tolist())), unit='Angstrom',
                 basis=basis, charge=0, spin=0, symmetry=False, cart=False, verbose=0)


def rhf(symbols, coords, basis='sto-3g', dm0=None):
    mf = scf.RHF(make_molecule(symbols, coords, basis))
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-13, 1e-10, 1e-14
    mf.max_cycle, mf.diis_space, mf.conv_check, mf.chkfile = 400, 12, False, None
    mf.kernel(dm0=dm0)
    if not mf.converged or np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)) > 1e-10:
        solver = mf.newton()
        solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-13, 1e-10, 400
        solver.kernel()
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = solver.mo_coeff, solver.mo_occ, solver.mo_energy
        mf.e_tot, mf.converged = solver.e_tot, solver.converged
    if not mf.converged or np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)) > 1e-10:
        raise RuntimeError('RHF did not converge to the specified orbital gradient tolerance.')
    return mf


def orbital_stability(mf):
    gradient, operator, _ = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ, with_symmetry=False)
    matrix = np.column_stack([2 * operator(v) for v in np.eye(gradient.size)])
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return 2 * gradient, values, vectors


def polish_orbitals(mf):
    previous = None
    for _ in range(30):
        energy = float(mf.energy_tot(dm=mf.make_rdm1()))
        residual = np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))
        if residual <= 1e-11 and previous is not None and abs(energy - previous) <= 1e-13:
            mf.e_tot, mf.converged = energy, True
            return True
        previous = energy
        if residual <= 1e-11:
            continue
        gradient, values, vectors = orbital_stability(mf)
        keep = abs(values) > 2e-10
        step = -vectors[:, keep] @ ((vectors[:, keep].T @ gradient) / values[keep])
        step *= min(1., .1 / max(float(np.linalg.norm(step)), 1e-30))
        mf.mo_coeff = mf.mo_coeff @ expm(scf.hf.unpack_uniq_var(step, mf.mo_occ))
    mf.e_tot, mf.converged = float(mf.energy_tot(dm=mf.make_rdm1())), False
    return False


def newton_restart(mf, orbitals=None):
    solver = mf.newton()
    solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-13, 1e-12, 400
    solver.ah_conv_tol, solver.ah_lindep = 1e-14, 1e-20
    solver.kernel(mo_coeff=mf.mo_coeff if orbitals is None else orbitals, mo_occ=mf.mo_occ)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = solver.mo_coeff, solver.mo_occ, solver.mo_energy
    mf.e_tot, mf.converged = solver.e_tot, solver.converged


def initial_rhf(symbols, coords, basis='sto-3g'):
    mf = scf.RHF(make_molecule(symbols, coords, basis))
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-13, 1e-12, 1e-14
    mf.max_cycle, mf.diis_space, mf.conv_check, mf.chkfile = 400, 12, False, None
    mf.direct_scf = False
    mf.kernel()
    good = polish_orbitals(mf)
    if not good:
        newton_restart(mf)
        good = polish_orbitals(mf)
    stable = False
    for repair in range(9):
        if not good:
            break
        _, values, vectors = orbital_stability(mf)
        stable = values[0] >= -1e-7
        if stable or repair == 8:
            break
        trials = [mf.mo_coeff @ expm(scf.hf.unpack_uniq_var(sign * .35 * vectors[:, 0], mf.mo_occ))
                  for sign in (-1., 1.)]
        orbitals = min(trials, key=lambda c: float(mf.energy_tot(dm=mf.make_rdm1(c, mf.mo_occ))))
        newton_restart(mf, orbitals)
        good = polish_orbitals(mf)
    if not good or not stable or np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)) > 1e-10:
        raise RuntimeError('The initial RHF state did not reach a stable, converged restricted solution.')
    return mf


def stretch(name, symbols, reference, masses, q):
    coords = reference.copy()
    if name in DIATOMICS:
        axis = reference[1] - reference[0]
        length = np.linalg.norm(axis)
        axis = axis / length
        coords[1] = coords[0] + (length + q) * axis
    elif name in STARS:
        heavy = symbols.index(STARS[name])
        for h, symbol in enumerate(symbols):
            if symbol == 'H':
                axis = reference[h] - reference[heavy]
                length = np.linalg.norm(axis)
                axis = axis / length
                coords[h] = coords[heavy] + (length + q) * axis
    else:
        o1, o2 = [i for i, symbol in enumerate(symbols) if symbol == 'O']
        hydrogens = [i for i, symbol in enumerate(symbols) if symbol == 'H']
        h1, h2 = min(permutations(hydrogens),
                     key=lambda p: sum(np.linalg.norm(reference[o] - reference[h])
                                       for o, h in zip((o1, o2), p)))
        axis = reference[o2] - reference[o1]
        length = np.linalg.norm(axis)
        axis = axis / length
        coords[o2] = coords[o1] + (length + q) * axis
        coords[h1] = coords[o1] + (reference[h1] - reference[o1])
        coords[h2] = coords[o2] + (reference[h2] - reference[o2])
    center = np.sum(masses[:, None] * reference, axis=0) / masses.sum()
    x = coords - np.sum(masses[:, None] * coords, axis=0) / masses.sum()
    y = reference - center
    u, _, vt = np.linalg.svd(x.T @ (masses[:, None] * y))
    proper = np.eye(3)
    proper[-1, -1] = 1. if np.linalg.det(u @ vt) >= 0 else -1.
    return x @ (u @ proper @ vt) + center


def equilibrium_scan(name, gtol=1e-6, maxiter=200):
    candidates = []
    for symbols, seed in starting_geometries(name):
        def objective(flat):
            mf = rhf(symbols, flat.reshape(-1, 3) * BOHR_A)
            return float(mf.e_tot), mf.nuc_grad_method().kernel().ravel()
        seed = seed - seed.mean(axis=0)
        result = minimize(objective, (seed / BOHR_A).ravel(), method='BFGS', jac=True,
                          options={'gtol': gtol, 'maxiter': maxiter})
        energy, gradient = float(result.fun), result.jac
        if result.success and np.max(abs(gradient)) <= gtol:
            candidates.append((energy, symbols, result.x.reshape(-1, 3) * BOHR_A))
    if not candidates:
        raise RuntimeError(f'{name}: no starting geometry reached the RHF equilibrium tolerance.')
    energy, symbols, reference = min(candidates, key=lambda c: c[0])

    reference = np.array([[float(f'{x:.14f}') for x in row] for row in reference])
    mf = rhf(symbols, reference)
    masses = mf.mol.atom_mass_list(isotope_avg=True)
    delta = 2e-4
    tangent = (stretch(name, symbols, reference, masses, delta)
               - stretch(name, symbols, reference, masses, -delta)) / (2 * delta)
    mf.conv_tol_cpscf = 1e-12
    hessian = mf.Hessian()
    hessian.max_cycle = 100
    h = hessian.kernel()

    modes = thermo.harmonic_analysis(mf.mol, h, mass=masses, imaginary_freq=False)
    if np.any(modes['freq_wavenumber'] < 0):
        raise RuntimeError(f'{name}: the RHF equilibrium has an imaginary vibrational mode')

    # The Hessian is in Ha/Bohr^2
    # The scan tangent is dimensionless
    curvature = float(np.einsum('ax,abxy,by->', tangent, h, tangent))
    mass = float(np.einsum('a,ax,ax->', masses, tangent, tangent))
    omega = math.sqrt(curvature * nist.HARTREE2J / (BOHR_A * 1e-10)**2
                      / (mass * nist.ATOMIC_MASS))
    oscillator_length = math.sqrt(nist.HBAR / (mass * nist.ATOMIC_MASS * omega)) / 1e-10
    width = math.sqrt(3) * oscillator_length
    if name in DIATOMICS:
        length = float(np.linalg.norm(reference[1] - reference[0]))
    elif name in STARS:
        heavy = symbols.index(STARS[name])
        length = float(np.mean([np.linalg.norm(reference[h] - reference[heavy])
                                for h, symbol in enumerate(symbols) if symbol == 'H']))
    else:
        oxygen = [i for i, symbol in enumerate(symbols) if symbol == 'O']
        length = float(np.linalg.norm(reference[oxygen[1]] - reference[oxygen[0]]))
    if length <= width:
        raise RuntimeError(f'{name}: the harmonic interval includes a nonpositive bond length')
    summary = {'molecule': name, 'symbols': symbols, 'equilibrium_cartesian_A': reference.tolist(),
               'r_e_A': length, 'E_RHF_re_Ha': energy, 'k_Ha_per_Bohr2': curvature,
               'mu_eff_amu': mass, 'omega_rad_per_s': omega, 'Delta_q_n1_A': width,
               'n1_min_A': length - width, 'n1_max_A': length + width}
    records = []
    for index, value in enumerate(np.linspace(length - width, length + width, 30), 1):
        q = float(value - length)
        records.append({'molecule': name, 'point_index': index,
                        'geometry_id': f'{index:03d}', 'symbols': symbols,
                        'cartesian_A': stretch(name, symbols, reference, masses, q),
                        'bond_length_A': float(value), 'q_A': q})
    return records, summary


def initial_state(record):
    mf = initial_rhf(record['symbols'], np.asarray(record['cartesian_A']))
    mf.mo_energy, mf.mo_coeff = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    ncore, ncas, nelec = ACTIVE_SPACES[record['molecule']]
    mc = mcscf.CASCI(mf, ncas, nelec, ncore=ncore)
    mc.fcisolver.conv_tol, mc.fcisolver.max_cycle = 1e-12, 200
    mc.verbose = 0
    mc.check_sanity()
    # Keep the CI and active Hamiltonian in the canonical RHF orbital basis.
    h2 = mc.get_h2eff()
    h1, core = mc.get_h1eff()
    if h1.shape[0] != ncas:
        raise RuntimeError(f'Active space size error: nmo={mf.mo_coeff.shape[1]}, ncore={ncore}, ncas={ncas}')
    total, ci = mc.fcisolver.kernel(h1, h2, ncas, mc.nelecas, ci0=mc.ci,
                                   verbose=lib.logger.new_logger(mc),
                                   max_memory=max(400, mc.max_memory - lib.current_memory()[0]),
                                   ecore=core)
    total = float(total)
    if not mc.fcisolver.converged:
        raise RuntimeError('Frozen-core FCI did not converge')
    
    point = dict(record, basis='sto-3g', charge=0, spin=0,
                 E_RHF_Ha=float(mf.e_tot), E_FCI_Ha=total,
                 E_corr_Ha=total - float(mf.e_tot),
                 E_nuclear_Ha=float(mf.mol.energy_nuc()),
                 E_CAS_active_Ha=total - float(core), E_core_including_nuclear_Ha=float(core),
                 n_frozen_orbitals=ncore, n_frozen_electrons=2*ncore,
                 n_active_orbitals=ncas, n_active_electrons=nelec,
                 fci_convention='frozen-core', total_energy_includes_nuclear_repulsion=True)
    point['cartesian_A'] = np.asarray(point['cartesian_A']).tolist()
    arrays = dict(cartesian_A=point['cartesian_A'], symbols=point['symbols'],
                  mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy_Ha=mf.mo_energy,
                  rhf_density_ao=mf.make_rdm1(), overlap_ao=mf.get_ovlp(),
                  hcore_ao_Ha=mf.get_hcore(), fock_ao_Ha=mf.get_fock(),
                  molecule_pyscf_json=mf.mol.dumps(), E_RHF_Ha=mf.e_tot,
                  ci_vector=np.asarray(ci), active_h1_Ha=h1,
                  active_h2_packed_Ha=h2, core_energy_Ha=core,
                  E_FCI_Ha=total, E_corr_Ha=point['E_corr_Ha'])
    return point, arrays


def transform_rank4(G, C):
    return np.einsum('ap,bq,cr,ds,pqrs->abcd', C, C, C, C, G, optimize=True)


def correlated_gamma(t2, nmo, nelectron):
    nocc = nelectron // 2
    singles = np.zeros((nocc, nmo - nocc))
    phi = cisd.to_fcivec(cisd.amplitudes_to_cisdvec(1., singles, np.zeros_like(t2)),
                        nmo, nelectron, frozen=0)
    chi = cisd.to_fcivec(cisd.amplitudes_to_cisdvec(0., singles, t2),
                        nmo, nelectron, frozen=0)
    w = float(np.vdot(chi, chi).real)
    occupations = np.zeros(nmo)
    occupations[:nocc] = 2.
    P = np.diag(occupations)
    # Spin-summed Gamma[p,q,r,s] = <p^dagger r^dagger s q>.
    g0 = np.einsum('pq,rs->pqrs', P, P) - 0.5 * np.einsum('ps,rq->pqrs', P, P)
    _, gr = direct_spin1.make_rdm12(phi + chi, nmo, nelectron, reorder=True)
    return gr - w * g0


def descriptors(point, initial):
    mol = make_molecule(point['symbols'], np.asarray(point['cartesian_A']))
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-13, 1e-10, 1e-14
    mf.max_cycle, mf.diis_space, mf.conv_check = 400, 12, False
    mf.direct_scf, mf.chkfile = False, None
    mf.kernel(dm0=initial['rhf_density_ao'])
    if not mf.converged:
        raise RuntimeError('Descriptor RHF did not converge')
    mf.mo_energy, mf.mo_coeff = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    pt = mp2.RMP2(mf, frozen=0)
    e_mp2, t2 = pt.kernel()
    C, S, P = mf.mo_coeff, mf.get_ovlp(), mf.make_rdm1()
    F = mf.get_fock(dm=P)
    eigenvalues, U = np.linalg.eigh(S)
    half = (U * np.sqrt(eigenvalues)) @ U.T
    inverse = (U / np.sqrt(eigenvalues)) @ U.T

    # Lowdin orthogonalization acts in full AO space, before valence selection.
    P_L, F_L = half @ P @ half, inverse @ F @ inverse
    gamma_mo = correlated_gamma(t2, C.shape[1], mol.nelectron)
    gamma_ao = transform_rank4(gamma_mo, C)
    gamma0 = np.einsum('pq,rs->pqrs', P, P) - 0.5*np.einsum('ps,rq->pqrs', P, P)
    cumulant = gamma_ao - gamma0
    gamma_L, lambda_L = transform_rank4(gamma_ao, half), transform_rank4(cumulant, half)

    # Source AO indices in model-qubit order
    symbols, active = MOLECULES[point['molecule']]
    atoms = []
    for symbol in symbols:
        atoms.append(next(i for i, s in enumerate(point['symbols']) if s == symbol and i not in atoms))
    labels = mol.ao_labels()
    full_order = [i for a in atoms for i, label in enumerate(mol.ao_labels(fmt=False)) if label[0] == a]
    selected = np.asarray([full_order[i] for i in active])
    values = density_descriptors(P_L[np.ix_(selected, selected)], gamma_L, lambda_L, selected)
    metadata = dict(rdm_source=RDM_SOURCE, perturbative_order=2,
                    expectation='R0+R1+R_chichi-w*R0; no exact division by 1+w',
                    spin_convention='spin-summed spatial-orbital, closed-shell singlet',
                    index_ordering_MO=RDM_CONVENTION, index_ordering_AO=RDM_CONVENTION,
                    frozen_core=0, input_fci_frozen_core=point['n_frozen_orbitals'],
                    calculation_space='all occupied and virtual MOs; full AOs until final descriptor extraction',
                    formal_cumulant='Gamma_MP2_AO_consistent-Gamma0[P_HF]')
    return dict(values, molecule=point['molecule'], geometry_id=point['geometry_id'],
                symbols=point['symbols'], cartesian_A=point['cartesian_A'], ao_labels=labels,
                q_A=point['q_A'], bond_length_A=point['bond_length_A'],
                basis='sto-3g', charge=0, spin=0, electron_count=mol.nelectron,
                units='Angstrom', energy_units='Hartree', pyscf_version=pyscf.__version__,
                E_RHF_input=point['E_RHF_Ha'], E_FCI=point['E_FCI_Ha'],
                E_corr_input=point['E_FCI_Ha'] - point['E_RHF_Ha'],
                E_RHF_calculated=mf.e_tot, E_MP2_correlation=e_mp2,
                mo_coeff=C, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy,
                S_AO=S, S_half=half, S_minus_half=inverse,
                P_AO=P, F_AO=F, P_L=P_L, F_L=F_L,
                Gamma_MP2_MO_consistent=gamma_mo, Gamma_MP2_AO_consistent=gamma_ao,
                Gamma0_AO=gamma0, Lambda_AO=cumulant, Lambda_L=lambda_L,
                source_selected_ao_indices=selected,
                pair_indices=np.asarray(list(combinations(range(len(selected)), 2))),
                triple_indices=np.asarray(list(combinations(range(len(selected)), 3))),
                quadruple_indices=np.asarray(list(combinations(range(len(selected)), 4))),
                rdm_frozen_core=0, input_fci_frozen_core=point['n_frozen_orbitals'],
                rdm_source=RDM_SOURCE, rdm_convention=RDM_CONVENTION,
                rdm_metadata_json=json.dumps(metadata))


def write_initial(folder, record):
    point, arrays = initial_state(record)
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'electronic_state.npz', **arrays)
    (folder / 'point.json').write_text(json.dumps(point, indent=2) + '\n')
    lines = [str(len(point['symbols'])), f"{point['molecule']} RHF/FCI STO-3G; Angstrom"]
    lines += [f'{s} {r[0]:.14f} {r[1]:.14f} {r[2]:.14f}'
              for s, r in zip(point['symbols'], point['cartesian_A'])]
    (folder / 'geometry.xyz').write_text('\n'.join(lines) + '\n')
    return point, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--molecules', nargs='+', choices=list(MOLECULES), default=list(MOLECULES))
    parser.add_argument('--stage', choices=('all', 'initial', 'descriptors'), default='all')
    parser.add_argument('--geometry-source', type=Path,
                        help='Reuse scan coordinates from an initialdata directory; recalculate RHF/FCI')
    parser.add_argument('--input-dir', type=Path, default=PROJECT / 'results' / 'initialdata',
                        help='Initial-state directory for --stage descriptors')
    parser.add_argument('--geometry-ids', nargs='+', type=int, choices=range(1, 31),
                        help='Calculate selected IDs from the unchanged 30-point grid')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    output = args.output_dir or PROJECT / 'results_paper_v2' / datetime.now().strftime('data_%Y%m%d_%H%M%S_%f')
    if output.resolve().is_relative_to(Path(__file__).resolve().parent):
        parser.error('Choose an output directory outside paper_source_v2')
    output.mkdir(parents=True, exist_ok=False)
    lib.num_threads(args.threads)
    for name in args.molecules:
        if args.stage == 'descriptors' or args.geometry_source is not None:
            source = args.input_dir if args.stage == 'descriptors' else args.geometry_source
            records = []
            for index in args.geometry_ids or range(1, 31):
                point = json.loads((source / name / 'scan' / f'{index:03d}' / 'point.json').read_text())
                if args.stage == 'descriptors':
                    point.update(molecule=name, geometry_id=f'{index:03d}')
                    records.append(point)
                else:
                    records.append(dict(molecule=name, geometry_id=f'{index:03d}', point_index=index,
                                        symbols=point['symbols'], cartesian_A=point['cartesian_A'],
                                        q_A=point['q_A'], bond_length_A=point['bond_length_A']))
        else:
            print(f'{name}: RHF equilibrium and harmonic n=1 scan range', flush=True)
            records, summary = equilibrium_scan(name)
            folder = output / 'initialdata' / name
            folder.mkdir(parents=True)
            (folder / 'scan_definition.json').write_text(json.dumps(summary, indent=2) + '\n')
            write_initial(folder / 'equilibrium_energies', dict(
                molecule=name, geometry_id='000', point_index=0, symbols=summary['symbols'],
                cartesian_A=summary['equilibrium_cartesian_A'], q_A=0., bond_length_A=summary['r_e_A']))
            if args.geometry_ids is not None:
                records = [r for r in records if r['point_index'] in args.geometry_ids]
        for record in records:
            gid = record['geometry_id']
            print(f'{name}/{gid}: {args.stage}', flush=True)
            if args.stage == 'descriptors':
                folder = args.input_dir / name / 'scan' / gid
                point = record
                with np.load(folder / 'electronic_state.npz', allow_pickle=False) as saved:
                    initial = {'rhf_density_ao': saved['rhf_density_ao']}
            else:
                folder = output / 'initialdata' / name / 'scan' / gid
                point, initial = write_initial(folder, record)
            if args.stage != 'initial':
                arrays = descriptors(point, initial)
                arrays['input_state_path'] = str((folder / 'electronic_state.npz').resolve())
                destination = output / 'descriptor' / name
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination / f'{gid}.npz', **arrays)
    print(f'Data: {output}', flush=True)


if __name__ == '__main__':
    main()
