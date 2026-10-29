"""Calculate descriptors after a full-AO Lowdin transformation.

All electrons and STO-3G orbitals participate in MP2/RDM construction.
Descriptor selection excludes heavy-atom 1s AOs.
The training loader derives model features from the saved full-AO tensors.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from .registry import MOLECULE_NAMES as MOLECULES, DEFAULT_INITIAL_DIR, DEFAULT_DESCRIPTOR_DIR

PROJECT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "paper-source-v5-native-mp2-v1"
NATIVE_SCHEMA_VERSION = "descriptor-v5-native-mp2"
RDM_SOURCE = "PySCF RMP2.make_rdm1/make_rdm2, unrelaxed, frozen=0"
RDM_CONVENTION = "Gamma[p,q,r,s]=sum_spin <p^dagger r^dagger s q>; P[p,q]=sum_spin <q^dagger p>"
DESCRIPTOR_LAYOUT = "full-rank4-no-triple-cache-v1"
AO_SELECTION_RULE = "post-lowdin-heavy-1s-excluded-v1"
RDM_SOURCES = {name: RDM_SOURCE for name in MOLECULES}


def load_records(input_dir, molecules):
    records = []
    for name in molecules:
        paths = sorted((input_dir / name / "scan").glob("*/point.json"), key=lambda p: int(p.parent.name))
        for path in paths:
            point = json.loads(path.read_text())
            records.append({**point, "geometry_id": path.parent.name,
                            "input_geometry_path": str(path.with_name("geometry.xyz").resolve()),
                            "input_state_path": str(path.with_name("electronic_state.npz").resolve())})
    return records


def disconnected_rhf(P):
    return np.einsum("pq,rs->pqrs", P, P) - .5*np.einsum("ps,rq->pqrs", P, P)


def lowdin_factors(S):
    eigenvalues, U = np.linalg.eigh(S)
    return (U*np.sqrt(eigenvalues))@U.T, (U/np.sqrt(eigenvalues))@U.T


def transform_rank4(tensor, A):
    return np.einsum("ap,bq,cr,ds,pqrs->abcd", A, A, A, A, tensor, optimize=True)


def select_valence_aos(mol):
    labels = mol.ao_labels()
    structured = mol.ao_labels(fmt=False)
    core, valence, shells = [], [], []
    locations = mol.ao_loc_nr()
    for shell in range(mol.nbas):
        atom = mol.bas_atom(shell)
        Z = mol.atom_charge(atom)
        for i in range(int(locations[shell]), int(locations[shell+1])):
            principal = int(structured[i][2][:-1])
            is_core = Z > 2 and principal == 1
            (core if is_core else valence).append(i)
            shells.append({"ao_index": i, "atom_index": atom, "shell_index": shell,
                           "principal_shell": principal, "angular_momentum": mol.bas_angular(shell),
                           "n_contractions": mol.bas_nctr(shell), "n_primitives": mol.bas_nprim(shell)})
    return {"ao_selection_rule": np.asarray(AO_SELECTION_RULE),
            "ao_labels": np.asarray(labels), "core_indices": np.asarray(core, dtype=int),
            "valence_indices": np.asarray(valence, dtype=int),
            "core_labels": np.asarray([labels[i] for i in core]),
            "valence_labels": np.asarray([labels[i] for i in valence]),
            "ao_shell_metadata_json": np.asarray(json.dumps(shells))}


def extract_descriptors(P_L, F_L, valence_indices):
    v = np.asarray(valence_indices)
    n = len(v)
    P, F = P_L[np.ix_(v,v)], F_L[np.ix_(v,v)]
    pairs = np.asarray(list(itertools.combinations(range(n),2)), dtype=int).reshape(-1,2)
    triples = np.asarray(list(itertools.combinations(range(n), 3)), dtype=int).reshape(-1, 3)
    return {"P_mu": np.diag(P).copy(), "F_mu": np.diag(F).copy(),
            "pair_indices": pairs, "P_munu": P[pairs[:,0],pairs[:,1]],
            "triple_indices": triples}


def calculate_record(record):
    import pyscf
    from pyscf import gto
    from pyscf.scf.hf import RHF
    from pyscf.mp import mp2
    if record["molecule"] not in MOLECULES:
        raise ValueError("Unsupported molecule")
    mol = gto.M(atom=list(zip(record["symbols"], record["cartesian_A"])), unit="Angstrom",
                basis=record["basis"], charge=record["charge"], spin=record["spin"],
                cart=False, symmetry=False, verbose=0)
    with np.load(record["input_state_path"], allow_pickle=False) as saved:
        dm0 = saved["rhf_density_ao"]
    mf = RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-13, 1e-10, 1e-14
    mf.max_cycle, mf.diis_space, mf.conv_check = 400, 12, False
    mf.direct_scf, mf.chkfile = False, None
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise ValueError("RHF did not converge; refusing to save a production descriptor")
    mf.mo_energy, mf.mo_coeff = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    pt = mp2.RMP2(mf, frozen=0)
    e_mp2, t2 = pt.kernel()
    C, S, P = mf.mo_coeff, mf.get_ovlp(), mf.make_rdm1()
    F = mf.get_fock(dm=P)
    arrays = {"S_AO": S, "P_AO": P, "F_AO": F,
              "hcore_AO": mf.get_hcore(), "eri_AO": mol.intor("int2e", aosym="s1"),
              "E_nuclear": mol.energy_nuc(), "E_RHF_calculated": mf.e_tot,
              "E_MP2_correlation": e_mp2, "E_MP2_total": mf.e_tot+e_mp2,
              "rhf_converged": mf.converged, "mp2_t2": t2,
              "mo_coeff": C, "mo_occ": mf.mo_occ, "mo_energy": mf.mo_energy,
              "P_MP2_MO_pyscf": pt.make_rdm1(ao_repr=False),
              "Gamma_MP2_MO_pyscf": pt.make_rdm2(ao_repr=False)}
    arrays['P_MP2_AO_pyscf'] = C @ arrays['P_MP2_MO_pyscf'] @ C.T
    arrays['Gamma_MP2_AO_pyscf'] = transform_rank4(arrays['Gamma_MP2_MO_pyscf'], C)
    gamma0 = disconnected_rhf(P)
    arrays.update(Gamma0_HF_AO=gamma0, Gamma0_AO=gamma0)
    half, inverse = lowdin_factors(S)
    arrays.update(S_half=half, S_minus_half=inverse, P_L=half@P@half,
                  F_L=inverse@F@inverse)
    arrays.update(select_valence_aos(mol))
    arrays.update(extract_descriptors(arrays["P_L"], arrays["F_L"], arrays["valence_indices"]))
    metadata = {"rdm_source": RDM_SOURCE, "perturbative_order": 2,
                "orbital_response": "unrelaxed; occupied-virtual response omitted",
                "model_one_body_density": "P_AO and P_L remain RHF",
                "spin_convention": "spin-summed spatial-orbital, closed-shell singlet",
                "index_ordering_MO": RDM_CONVENTION, "index_ordering_AO": RDM_CONVENTION,
                "frozen_core": 0, "input_fci_frozen_core": record["n_frozen_orbitals"],
                "calculation_space": "all occupied and virtual MOs; full AOs until final descriptor extraction",
                "pyscf_version": pyscf.__version__}
    arrays.update(molecule=record["molecule"], geometry_id=record["geometry_id"],
                  point_index=record["point_index"], input_geometry_path=record["input_geometry_path"],
                  symbols=record["symbols"], cartesian_A=record["cartesian_A"],
                  input_state_path=record["input_state_path"], scan_coordinate=record["q_A"],
                  q_A=record["q_A"], bond_length_A=record["bond_length_A"], units="Angstrom", energy_units="Hartree",
                  charge=mol.charge, spin=mol.spin, basis=record["basis"], electron_count=mol.nelectron,
                  frozen_core=0, mp2_frozen_core=0, rdm_frozen_core=0,
                  input_fci_frozen_core=record["n_frozen_orbitals"], input_fci_convention=record["fci_convention"],
                  E_RHF_input=record["E_RHF_Ha"], E_FCI=record["E_FCI_Ha"], E_corr_input=record["E_corr_Ha"],
                  E_FCI_minus_E_RHF_input=record["E_FCI_Ha"]-record["E_RHF_Ha"],
                  schema_version=NATIVE_SCHEMA_VERSION,
                  descriptor_layout=DESCRIPTOR_LAYOUT, rdm_source=RDM_SOURCE,
                  rdm_convention=RDM_CONVENTION,
                  rdm_metadata_json=json.dumps(metadata), pyscf_version=pyscf.__version__)
    return {key: np.asarray(value) for key,value in arrays.items()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def save_record(output, record, arrays=None, error=None):
    base = output/record["molecule"]/record["geometry_id"]
    if base.with_suffix(".npz").exists():
        raise FileExistsError(f"Descriptor already exists: {base.with_suffix('.npz')}")
    base.parent.mkdir(parents=True, exist_ok=True)
    if base.with_suffix(".json").exists():
        attempt = 1
        while base.with_suffix(f".attempt{attempt}.json").exists():
            attempt += 1
        base.with_suffix(".json").replace(base.with_suffix(f".attempt{attempt}.json"))
    row = {"molecule": record["molecule"], "geometry_id": record["geometry_id"],
           "q_A": record["q_A"],
           "record": str(base.with_suffix(".npz")) if arrays is not None else "", "error": error or ""}
    if arrays is not None:
        with base.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        base.with_suffix(".npz.tmp").replace(base.with_suffix(".npz"))
    write_json(base.with_suffix(".json"), row)
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", "--source-dir", dest="input_dir", type=Path,
                        help="Initial-state directory; relative paths use the project root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DESCRIPTOR_DIR,
                        help="Descriptor destination; relative paths use the project root")
    parser.add_argument("--molecules", nargs="+", choices=[*MOLECULES, "all"])
    parser.add_argument("--geometry-id", help="Single scan ID; use with one molecule")
    parser.add_argument("--smoke-test", action="store_true", help="Calculate H2O/001 only")
    parser.add_argument("--dry-run", action="store_true", help="List selected input records only")
    parser.add_argument("--resume", action="store_true", help="Skip completed files")
    parser.add_argument("--rerun-failed", action="store_true", help="Retry execution failures; keep completed files")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    args.input_dir = args.input_dir if args.input_dir is not None else DEFAULT_INITIAL_DIR
    for name in ("input_dir", "output_dir"):
        path = getattr(args, name).expanduser()
        setattr(args, name, (path if path.is_absolute() else PROJECT/path).resolve())
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.molecules = list(MOLECULES) if args.molecules is None or args.molecules == ["all"] else args.molecules
    if "all" in args.molecules or len(set(args.molecules)) != len(args.molecules):
        parser.error("Use all alone or select unique molecule IDs")
    if args.smoke_test:
        args.molecules, args.geometry_id = ["H2O"], "001"
    if args.geometry_id:
        if len(args.molecules) != 1:
            parser.error("--geometry-id requires one molecule")
        if not args.geometry_id.isdigit() or not 1 <= int(args.geometry_id) <= 30:
            parser.error("--geometry-id must be between 001 and 030")
        args.geometry_id = f"{int(args.geometry_id):03d}"
    return args


def run(args):
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Input and output directories must be separate and non-nested")
    records = load_records(source, args.molecules)
    if args.geometry_id:
        records = [r for r in records if r["geometry_id"] == args.geometry_id]
    missing = set(args.molecules) - {r["molecule"] for r in records}
    if missing:
        raise FileNotFoundError(f"No matching scan records for: {', '.join(sorted(missing))}")
    for record in records:
        for key in ("input_geometry_path", "input_state_path"):
            if not Path(record[key]).is_file():
                raise FileNotFoundError(record[key])
    if args.dry_run:
        print(json.dumps({"geometry_count": len(records), "records": [
            {k:r[k] for k in ("molecule", "geometry_id", "input_geometry_path")} for r in records]}, indent=2))
        return 0
    if output.exists() and any(output.iterdir()) and not (args.resume or args.rerun_failed):
        raise FileExistsError("Output exists; use --resume or --rerun-failed, or a new output directory")
    manifest_path = output/"manifest.json"
    if output.exists() and any(output.iterdir()) and not manifest_path.is_file():
        raise ValueError("Existing output has no manifest; use a new output directory")
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"records": []}
    if previous["records"] and previous.get("ao_selection_rule") != AO_SELECTION_RULE:
        raise ValueError("Existing output uses an older or unknown post-Lowdin AO selection; "
                         "use a new output directory for the current AO selection.")
    if previous["records"] and previous.get("descriptor_layout") != DESCRIPTOR_LAYOUT:
        raise ValueError("Existing output uses an older or unknown descriptor layout; "
                         "use a new output directory without historical triple caches.")
    if previous["records"] and Path(previous.get("input_dir", "")).resolve() != source:
        raise ValueError("Existing output uses a different input directory; use a new output directory")
    from pyscf import lib
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"OpenMP is not available\. Setting omp_threads to .* has no effects\.",
                                category=UserWarning, module=r"pyscf\.lib\.misc")
        lib.num_threads(args.threads)
    entries = {(r["molecule"],r["geometry_id"]): r for r in previous["records"]}
    fields = ["molecule", "geometry_id", "q_A", "record", "error"]
    def checkpoint():
        rows = sorted(entries.values(), key=lambda r: (MOLECULES.index(r["molecule"]),int(r["geometry_id"])))
        manifest = {"schema_version": SCHEMA_VERSION,
                    "descriptor_layout": DESCRIPTOR_LAYOUT,
                    "ao_selection_rule": AO_SELECTION_RULE,
                    "rdm_sources": RDM_SOURCES, "mp2_frozen_core": 0,
                    "input_dir": str(source), "records": rows,
                    "completed_geometry_count": sum(bool(r.get("record")) and not r.get("error") for r in rows),
                    "failed_geometry_count": sum(bool(r.get("error")) for r in rows)}
        write_json(manifest_path, manifest)
        for name in sorted({r["molecule"] for r in rows}):
            with (output/name/"summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(r for r in rows if r["molecule"] == name)
    exit_code = 0
    for record in records:
        key = record["molecule"],record["geometry_id"]
        base = output/key[0]/key[1]
        sidecar = json.loads(base.with_suffix(".json").read_text()) if base.with_suffix(".json").exists() else {}
        failed = bool(sidecar.get("error")) or sidecar.get("status") in ("FAILED", "FAIL")
        if base.with_suffix(".npz").exists() and failed:
            raise FileExistsError(f"Existing NPZ has failed sidecar metadata; use a new output directory: {base}")
        if base.with_suffix(".npz").exists():
            entries[key] = {"molecule": key[0], "geometry_id": key[1], "q_A": record["q_A"],
                            "record": str(base.with_suffix(".npz")), "error": ""}
            print(f"SKIP {key[0]}/{key[1]}", flush=True)
            continue
        if failed and not args.rerun_failed:
            entries[key] = {k:sidecar[k] for k in fields}
            print(f"FAILED {key[0]}/{key[1]}: use --rerun-failed to retry", flush=True)
            exit_code = 1
            continue
        try:
            arrays = calculate_record(record)
            entries[key] = save_record(output, record, arrays)
        except Exception as exc:
            entries[key] = save_record(output, record, error=f"{type(exc).__name__}: {exc}")
            exit_code = 1
        checkpoint()
        print(f"{key[0]}/{key[1]} {entries[key]['error'] or entries[key]['record']}", flush=True)
    checkpoint()
    print(f"Completed: {sum(bool(r.get('record')) and not r.get('error') for r in entries.values())}; execution failures: "
          f"{sum(bool(r.get('error')) for r in entries.values())}.", flush=True)
    return exit_code


def main(argv=None):
    try:
        args = parse_args(argv)
        return run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
