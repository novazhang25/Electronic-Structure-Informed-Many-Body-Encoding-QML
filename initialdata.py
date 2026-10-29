"""Standalone RHF equilibrium, harmonic n=1 range, RHF/FCI scan and correlation.
Default: six neutral singlets, STO-3G, 30 inclusive scan points.
FCI uses the project's frozen-core convention on canonical RHF orbitals.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import itertools
import json
import math
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.optimize import brentq, minimize
import pyscf
from pyscf import gto, lib, mcscf, scf
from pyscf.data import nist
from pyscf.hessian import thermo
from pyscf.soscf import newton_ah

from .registry import MOLECULE_NAMES as MOLECULES, DEFAULT_INITIAL_DIR

PROJECT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "initialdata-v1"
ACTIVE_SPACES = {
    "HF": (1, 5, 8), "H2O": (1, 6, 8), "NH3": (1, 7, 8),
    "N2": (2, 8, 10), "CO": (2, 8, 10), "H2O2": (2, 10, 14),
}
SCF_GRADIENT_LIMIT = 1e-10
STABILITY_LIMIT = -1e-7


FORMULAS = {
    "HF": {"H": 1, "F": 1}, "H2O": {"O": 1, "H": 2},
    "NH3": {"N": 1, "H": 3}, "N2": {"N": 2},
    "CO": {"C": 1, "O": 1}, "H2O2": {"O": 2, "H": 2},
}
STARS = {"H2O": "O", "NH3": "N"}
DIATOMICS = {"N2", "CO", "HF"}
BOHR_A = nist.BOHR
BOHR_M = BOHR_A * 1e-10
EH_J = nist.HARTREE2J
AMU_KG = nist.ATOMIC_MASS
HBAR = nist.HBAR
C_MS = nist.LIGHT_SPEED_SI
TANGENT_STEPS = (1e-3, 5e-4, 2e-4)
ENERGY_STEPS = (0.01, 0.005, 0.002, 0.001)
TOL = {
    "gradient_max_Eh_per_Bohr": 2e-6,
    "gradient_rms_Eh_per_Bohr": 1e-6,
    "common_bond_spread_A": 1e-6,
    "bond_derivative": 2e-8,
    "fixed_length_error_A": 2e-10,
    "fixed_angle_error_deg": 2e-7,
    "fixed_angle_derivative_deg_per_A": 2e-5,
    "tangent_convergence_max": 2e-7,
    "com_derivative": 2e-9,
    "rotation_residual_amu_A": 2e-7,
    "alignment_distance_error_A": 2e-12,
    "hessian_symmetry_Eh_per_Bohr2": 1e-8,
    "hessian_translation_Eh_per_Bohr2": 1e-7,
    "hessian_gradient_probe_relative": 2e-5,
    "hessian_gradient_probe_absolute_Eh_per_Bohr2": 2e-7,
    "fd_relative": 2e-4,
    "fd_absolute_Eh_per_Bohr2": 2e-6,
    "mode_orthogonality": 2e-7,
    "mode_weight_sum": 2e-7,
    "mode_curvature_relative": 2e-7,
    "diatomic_mass_relative": 2e-8,
}


class InputProblem(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_equilibrium(name, row, metadata_path):
    if row is None:
        raise InputProblem("No equilibrium metadata row is available.")
    if row.get("method", "").strip().upper() != "RHF":
        raise InputProblem("Metadata must establish method=RHF.")
    basis = row.get("basis", "").strip()
    if not basis:
        raise InputProblem("The equilibrium RHF basis is missing.")
    if row.get("status", "").lower() != "converged":
        raise InputProblem("Equilibrium optimization is incomplete; provide a converged structure.")
    if row.get("optimizer_success", "true").strip().lower() not in ("true", "1", "yes"):
        raise InputProblem("Equilibrium optimizer did not report success.")
    xyz_string = row.get("xyz_file", "").strip()
    if not xyz_string:
        raise InputProblem("The optimized Cartesian geometry path is missing.")
    supplied_path = Path(xyz_string).expanduser()
    xyz_path = supplied_path if supplied_path.is_absolute() else metadata_path.parent / supplied_path

    if not xyz_path.is_file() and supplied_path.is_absolute():
        xyz_path = metadata_path.parent / supplied_path.name
    if not xyz_path.is_file():
        raise InputProblem(f"Optimized XYZ file is missing: {xyz_path}")
    lines = xyz_path.read_text().splitlines()
    try:
        n = int(lines[0])
        records = [line.split() for line in lines[2:2 + n]]
        symbols = [r[0] for r in records]
        coords = np.array([[float(v) for v in r[1:4]] for r in records])
    except (ValueError, IndexError) as exc:
        raise InputProblem(f"Invalid optimized XYZ: {exc}") from exc
    if coords.shape != (n, 3) or not np.isfinite(coords).all():
        raise InputProblem("XYZ coordinates are incomplete or nonfinite.")
    if Counter(symbols) != Counter(FORMULAS[name]):
        raise InputProblem(f"XYZ atom composition does not match {name}.")
    comment = lines[1]
    match = re.search(r"\bRHF/(\S+)", comment, flags=re.I)
    if match and match.group(1).lower() != basis.lower():
        raise InputProblem("XYZ comment and metadata disagree on the RHF basis.")
    return symbols, coords, basis, {
        "metadata_file": str(metadata_path.resolve()), "metadata_sha256": sha256(metadata_path),
        "xyz_file": str(xyz_path.resolve()), "xyz_sha256": sha256(xyz_path),
        "xyz_comment": comment, "metadata_row": dict(row),
        "geometry_units": "Angstrom", "reoptimized": False,
    }


def bond(coords, i, j):
    return float(np.linalg.norm(coords[j] - coords[i]))


def angle(coords, i, j, k):
    a, b = coords[i] - coords[j], coords[k] - coords[j]
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def dihedral(coords, i, j, k, l):
    axis = coords[k] - coords[j]
    axis = axis / np.linalg.norm(axis)
    v, w = coords[i] - coords[j], coords[l] - coords[k]
    v, w = v - np.dot(v, axis) * axis, w - np.dot(w, axis) * axis
    if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-10:
        raise ValueError("The H-O-O-H dihedral is undefined for this geometry.")
    return math.degrees(math.atan2(np.dot(np.cross(axis, v), w), np.dot(v, w)))


def angular_difference(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def center(coords, masses):
    return np.sum(masses[:, None] * coords, axis=0) / masses.sum()


def align(coords, reference, masses):
    ref_com = center(reference, masses)
    x, y = coords - center(coords, masses), reference - ref_com
    u, _, vt = np.linalg.svd(x.T @ (masses[:, None] * y))
    proper = np.eye(3)
    proper[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ proper @ vt
    return x @ rotation + ref_com, rotation


class StretchPath:
    def __init__(self, name, symbols, reference, masses):
        self.name, self.symbols = name, list(symbols)
        self.ref, self.masses = np.array(reference, float), np.array(masses, float)
        self.stretches, self.fixed_bonds, self.angles, self.dihedrals = [], [], [], []
        if name in DIATOMICS:
            self.stretches = [(0, 1)]
            self.definition = f"q = r({symbols[0]}-{symbols[1]}) - r_e({symbols[0]}-{symbols[1]})"
        elif name in STARS:
            self.heavy = symbols.index(STARS[name])
            self.hydrogens = [i for i, sym in enumerate(symbols) if sym == "H"]
            self.stretches = [(self.heavy, h) for h in self.hydrogens]
            self.angles = [(i, self.heavy, j) for i, j in itertools.combinations(self.hydrogens, 2)]
            self.definition = f"r({STARS[name]}-H_i) = r_e({STARS[name]}-H_i) + q for every H"
        elif name == "H2O2":
            self.oxygens = [i for i, sym in enumerate(symbols) if sym == "O"]
            hydrogens = [i for i, sym in enumerate(symbols) if sym == "H"]
            assignments = sorted(
                (sum(bond(self.ref, o, h) for o, h in zip(self.oxygens, perm)), perm)
                for perm in itertools.permutations(hydrogens)
            )
            if abs(assignments[0][0] - assignments[1][0]) < 1e-8:
                raise ValueError("Ambiguous OH connectivity in the optimized peroxide geometry.")
            h1, h2 = assignments[0][1]
            o1, o2 = self.oxygens
            self.peroxide_h = (h1, h2)
            self.stretches = [(o1, o2)]
            self.fixed_bonds = [(o1, h1), (o2, h2)]
            self.angles = [(h1, o1, o2), (o1, o2, h2)]
            self.dihedrals = [(h1, o1, o2, h2)]
            self.definition = "q = r(O-O) - r_e(O-O); both OH lengths, OOH angles and HOOH torsion fixed"
        else:
            raise ValueError(f"Unsupported molecule: {name}")
        self.lengths = np.array([bond(self.ref, *pair) for pair in self.stretches])
        if np.any(self.lengths < 1e-8):
            raise ValueError("A participating equilibrium bond has zero length.")
        self.s_eq = float(np.mean(self.lengths))

    def raw(self, q):
        if np.any(self.lengths + q <= 0):
            raise ValueError("A stretch displacement would produce a nonpositive bond length.")
        coords = self.ref.copy()
        if self.name in DIATOMICS:
            axis = (self.ref[1] - self.ref[0]) / self.lengths[0]
            coords[1] = coords[0] + (self.lengths[0] + q) * axis
        elif self.name in STARS:
            for (heavy, h), length in zip(self.stretches, self.lengths):
                ray = (self.ref[h] - self.ref[heavy]) / length
                coords[h] = coords[heavy] + (length + q) * ray
        else:
            o1, o2 = self.oxygens
            h1, h2 = self.peroxide_h
            axis = (self.ref[o2] - self.ref[o1]) / self.lengths[0]
            coords[o2] = coords[o1] + (self.lengths[0] + q) * axis
            coords[h1] = coords[o1] + (self.ref[h1] - self.ref[o1])
            coords[h2] = coords[o2] + (self.ref[h2] - self.ref[o2])
        return coords

    def __call__(self, q):
        coords_raw = self.raw(q)
        if self.name == "NH3":
            com_raw = np.average(coords_raw, axis=0, weights=self.masses)
            com_eq = np.average(self.ref, axis=0, weights=self.masses)
            return coords_raw + (com_eq - com_raw)
        return align(coords_raw, self.ref, self.masses)[0]

    def internals(self, coords):
        def label(indices):
            return "-".join(f"{self.symbols[i]}{i + 1}" for i in indices)
        values = {}
        for pairs, role in ((self.stretches, "stretch"), (self.fixed_bonds, "fixed_bond")):
            values.update({label(pair): {"value": bond(coords, *pair), "kind": role, "unit": "A"}
                           for pair in pairs})
        for indices in self.angles:
            values[label(indices)] = {"value": angle(coords, *indices), "kind": "fixed_angle", "unit": "deg"}
        for indices in self.dihedrals:
            values[label(indices)] = {"value": dihedral(coords, *indices), "kind": "fixed_dihedral", "unit": "deg"}
        return values

    def validate(self):
        samples, tangents, failures = [], [], []
        equilibrium = self.internals(self.ref)
        if np.ptp(self.lengths) > TOL["common_bond_spread_A"]:
            failures.append("Participating equilibrium lengths differ; a single common absolute s is undefined.")
        for delta in TANGENT_STEPS:
            plus, minus = self(delta), self(-delta)
            t = (plus - minus) / (2 * delta)
            tangents.append(t)
            residual_com = np.linalg.norm(center(t, self.masses))
            residual_rot = np.linalg.norm(np.sum(
                self.masses[:, None] * np.cross(self.ref - center(self.ref, self.masses), t), axis=0))
            p, m = self.internals(plus), self.internals(minus)
            derivatives, deviations = {}, {}
            for key, item in equilibrium.items():
                diff = angular_difference if item["kind"] == "fixed_dihedral" else lambda a, b: a - b
                derivative = diff(p[key]["value"], m[key]["value"]) / (2 * delta)
                derivatives[key] = derivative
                if item["kind"] == "stretch":
                    error = max(abs(p[key]["value"] - item["value"] - delta),
                                abs(m[key]["value"] - item["value"] + delta))
                    good = abs(derivative - 1) < TOL["bond_derivative"] and error < TOL["fixed_length_error_A"]
                else:
                    error = max(abs(diff(p[key]["value"], item["value"])),
                                abs(diff(m[key]["value"], item["value"])))
                    if item["unit"] == "A":
                        good = abs(derivative) < TOL["bond_derivative"] and error < TOL["fixed_length_error_A"]
                    else:
                        good = abs(derivative) < TOL["fixed_angle_derivative_deg_per_A"] and error < TOL["fixed_angle_error_deg"]
                deviations[key] = error
                if not good:
                    failures.append(f"Internal coordinate validation failed for {key} at delta={delta} A.")
            alignment_error, determinants = 0.0, []
            for q in (-delta, delta):
                raw = self.raw(q)
                if self.name == "NH3":
                    aligned = self(q)
                else:
                    aligned, rot = align(raw, self.ref, self.masses)
                    determinants.append(float(np.linalg.det(rot)))
                for i, j in itertools.combinations(range(len(self.ref)), 2):
                    alignment_error = max(alignment_error, abs(bond(raw, i, j) - bond(aligned, i, j)))
            if residual_com > TOL["com_derivative"] or residual_rot > TOL["rotation_residual_amu_A"]:
                failures.append(f"Translation/rotation residual too large at delta={delta} A.")
            if alignment_error > TOL["alignment_distance_error_A"] or not np.allclose(determinants, 1, atol=1e-12, rtol=0):
                failures.append("Alignment changed internal distances or was not a proper rotation.")
            samples.append({"delta_A": delta, "tangent": t, "internal_derivatives_per_A": derivatives,
                            "internal_max_deviations": deviations, "com_translation_residual": residual_com,
                            "rotation_residual_amu_A": residual_rot, "alignment_distance_error_A": alignment_error,
                            "rotation_determinants": determinants})
        convergence = [float(np.max(np.abs(t - tangents[-1]))) for t in tangents]
        if max(convergence) > TOL["tangent_convergence_max"]:
            failures.append("The Cartesian tangent did not converge with displacement step.")
        return tangents[-1], {"equilibrium_internals": equilibrium, "samples": samples,
                              "tangent_max_differences_from_finest": convergence, "failures": failures}


def make_molecule(symbols, coords, basis):
    mol = gto.M(atom=list(zip(symbols, coords.tolist())), unit="Angstrom", basis=basis,
                charge=0, spin=0, symmetry=False, cart=False, verbose=0)
    if mol.has_ecp() or mol.nelectron != sum(mol.atom_charges()):
        raise ValueError("This workflow requires a neutral all-electron molecule.")
    return mol


def run_rhf(symbols, coords, basis, dm0=None):
    mf = scf.RHF(make_molecule(symbols, coords, basis))
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-13, 1e-10, 1e-14
    mf.max_cycle, mf.diis_space = 400, 12
    mf.conv_check = False
    mf.chkfile = None
    mf.kernel(dm0=dm0)
    orbital_gradient = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    if not mf.converged or orbital_gradient > 1e-10:
        solver = mf.newton()
        solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-13, 1e-10, 400
        solver.kernel()
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = solver.mo_coeff, solver.mo_occ, solver.mo_energy
        mf.e_tot, mf.converged = solver.e_tot, solver.converged
        orbital_gradient = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    if not mf.converged or orbital_gradient > 1e-10 or not np.isfinite(mf.e_tot):
        raise RuntimeError(f"RHF SCF failed tight convergence; orbital gradient norm={orbital_gradient:.3e}")
    if getattr(mf, "with_df", None) is not None:
        raise RuntimeError("Density fitting is forbidden in this workflow.")
    return mf


def probability_mass(n, a):
    if n == 0:
        return math.erf(a)
    if n == 1:
        return math.erf(a) - 2 * a * math.exp(-a * a) / math.sqrt(math.pi)
    raise ValueError("Only n=0 and n=1 are supported.")


def oscillator_ranges(k_bohr, mu_amu, s_eq):
    k_si, mu_kg = k_bohr * EH_J / BOHR_M**2, mu_amu * AMU_KG
    if k_si <= 0 or mu_kg <= 0 or not np.isfinite([k_si, mu_kg]).all():
        raise ValueError("A positive finite curvature and generalized mass are required.")
    omega = math.sqrt(k_si / mu_kg)
    length = math.sqrt(HBAR / (mu_kg * omega)) / 1e-10
    n1 = math.sqrt(3) * length
    summary = {
        "k_q_Eh_per_Bohr2": k_bohr, "k_q_Eh_per_A2": k_bohr / BOHR_A**2,
        "k_q_N_per_m": k_si, "mu_eff_amu": mu_amu, "mu_eff_kg": mu_kg,
        "omega_rad_per_s": omega, "harmonic_frequency_cm1": omega / (2 * math.pi * C_MS * 100),
        "Delta_q_n0_A": length, "Delta_q_n1_A": n1,
        "n0_min_A": s_eq - length, "n0_max_A": s_eq + length,
        "n1_min_A": s_eq - n1, "n1_max_A": s_eq + n1,
    }
    probability_intervals = []
    for n in (0, 1):
        for probability in (0.95, 0.99, 0.999):
            a = brentq(lambda z: probability_mass(n, z) - probability, 0, 10, xtol=1e-14)
            width = a * length
            probability_intervals.append({"n": n, "probability": probability, "half_width_in_l": a,
                                          "q_interval_A": [-width, width],
                                          "s_interval_A": [s_eq - width, s_eq + width]})
    return summary, {
        "oscillator_length_A": length, "q_n0_minus_A": -length, "q_n0_plus_A": length,
        "q_n1_minus_A": -n1, "q_n1_plus_A": n1,
        "probability_intervals": probability_intervals,
    }


def normal_modes(mol, hessian, tangent, masses, k_bohr, mu_amu):
    analysis = thermo.harmonic_analysis(mol, hessian, mass=masses, imaginary_freq=False)
    modes = analysis["norm_mode"]
    weighted = modes * np.sqrt(masses)[None, :, None]
    flat = weighted.reshape(len(modes), -1)
    unit_t = (np.sqrt(masses)[:, None] * tangent / math.sqrt(mu_amu)).ravel()
    overlaps = flat @ unit_t
    weights = overlaps**2
    reconstruction = float(np.dot(weights, analysis["force_const_au"]))
    target = k_bohr / mu_amu
    reconstruction_error = abs(reconstruction - target) / max(abs(target), 1e-30)
    rows = [{"mode": i + 1, "frequency_cm1": float(analysis["freq_wavenumber"][i]),
             "signed_overlap": float(overlaps[i]), "overlap": float(abs(overlaps[i])),
             "squared_overlap_weight": float(weights[i])} for i in range(len(modes))]
    dominant = int(np.argmax(weights))
    return {
        "modes": rows, "cartesian_mass_normalized_modes": modes, "mode_indexing": "1-based; ascending eigenvalue",
        "dominant_normal_mode": dominant + 1,
        "dominant_mode_frequency_cm1": rows[dominant]["frequency_cm1"],
        "mode_overlap": rows[dominant]["overlap"], "sum_squared_weights": float(weights.sum()),
        "mixed_modes": bool(np.count_nonzero(weights > 1e-6) > 1),
        "resolved_mode_weight_threshold": 1e-6,
        "secondary_weight_sum": float(weights.sum() - weights[dominant]),
        "substantial_mixing": bool(weights[dominant] < 0.99),
        "substantial_mixing_threshold_dominant_weight": 0.99,
        "mass_orthogonality_max_error": float(np.max(np.abs(flat @ flat.T - np.eye(len(modes))))),
        "curvature_reconstruction_relative_error": reconstruction_error,
        "imaginary_mode_count": int(np.sum(analysis["freq_wavenumber"] < 0)),
    }


def calculate_harmonic(name, row, metadata_path):
    summary = {"molecule": name, "RHF_basis": (row or {}).get("basis", ""), "validation_status": "ERROR"}
    result = {"summary": summary, "failures": []}
    start = time.monotonic()
    try:
        symbols, reference, basis, provenance = load_equilibrium(name, row, metadata_path)
        result["input"] = provenance
        result["symbols"], result["equilibrium_cartesian_A"] = symbols, reference
        mol = make_molecule(symbols, reference, basis)
        masses = mol.atom_mass_list(isotope_avg=True)
        result["isotope_average_masses_amu"] = masses
        path = StretchPath(name, symbols, reference, masses)
        tangent, diagnostics = path.validate()
        result["path_validation"], result["tangent_dR_dq"] = diagnostics, tangent
        summary.update(RHF_basis=basis, coordinate_definition=path.definition, s_eq_A=path.s_eq)
        result["failures"].extend(diagnostics["failures"])
        if diagnostics["failures"]:
            summary["validation_status"] = "FAIL_PATH"
            return result

        mf = run_rhf(symbols, reference, basis)
        gradient = mf.nuc_grad_method().kernel()
        gradient_max = float(np.max(np.abs(gradient)))
        gradient_rms = float(np.sqrt(np.mean(gradient**2)))
        gradient_projection = float(np.sum(gradient * tangent))
        result["stationarity"] = {
            "E_RHF_Eh": float(mf.e_tot), "gradient_Eh_per_Bohr": gradient,
            "gradient_max_Eh_per_Bohr": gradient_max, "gradient_rms_Eh_per_Bohr": gradient_rms,
            "gradient_projection_Eh_per_Bohr": gradient_projection,
            "gradient_projection_Eh_per_A": gradient_projection / BOHR_A,
            "orbital_gradient_norm_Eh": float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))),
        }
        if gradient_max > TOL["gradient_max_Eh_per_Bohr"] or gradient_rms > TOL["gradient_rms_Eh_per_Bohr"]:
            result["failures"].append("FAIL_STATIONARITY: Cartesian nuclear gradient exceeds tolerance.")

        hessobj = mf.Hessian()
        hessobj.max_cycle = 100

        mf.conv_tol_cpscf = 1e-12
        hessian = hessobj.kernel()
        if hessian.shape != (mol.natm, mol.natm, 3, 3) or not np.isfinite(hessian).all():
            raise RuntimeError("Analytic Hessian has an invalid shape or nonfinite values.")
        symmetry_error = float(np.max(np.abs(hessian - hessian.transpose(1, 0, 3, 2))))
        translation_error = float(np.max(np.abs(hessian.sum(axis=1))))

        probe = np.sin(np.arange(1, mol.natm * 3 + 1)).reshape(mol.natm, 3)
        probe /= np.linalg.norm(probe)
        epsilon_bohr = 1e-4
        probe_gradients = []
        for sign in (1, -1):
            displaced = reference + sign * epsilon_bohr * BOHR_A * probe
            check_mf = run_rhf(symbols, displaced, basis, mf.make_rdm1())
            probe_gradients.append(check_mf.nuc_grad_method().kernel())
        numerical_response = (probe_gradients[0] - probe_gradients[1]) / (2 * epsilon_bohr)
        analytic_response = np.einsum("abxy,by->ax", hessian, probe)
        probe_error = float(np.max(np.abs(numerical_response - analytic_response)))
        probe_scale = float(np.max(np.abs(analytic_response)))
        result["hessian_validation"] = {
            "convention": "H[a,b,x,y] = d2E/(dR[a,x] dR[b,y]); Eh/Bohr^2",
            "shape": hessian.shape, "symmetry_max_error_Eh_per_Bohr2": symmetry_error,
            "translation_max_error_Eh_per_Bohr2": translation_error,
            "gradient_probe_step_Bohr": epsilon_bohr, "gradient_probe_vector": probe,
            "gradient_probe_max_error_Eh_per_Bohr2": probe_error,
            "gradient_probe_relative_error": probe_error / max(probe_scale, 1e-30),
            "analytic_gradient_response": analytic_response, "numerical_gradient_response": numerical_response,
        }
        result["cartesian_hessian_Eh_per_Bohr2"] = hessian
        if symmetry_error > TOL["hessian_symmetry_Eh_per_Bohr2"] or translation_error > TOL["hessian_translation_Eh_per_Bohr2"]:
            result["failures"].append("FAIL_HESSIAN: symmetry or translational sum rule failed.")
        if probe_error > TOL["hessian_gradient_probe_absolute_Eh_per_Bohr2"] + TOL["hessian_gradient_probe_relative"] * probe_scale:
            result["failures"].append("FAIL_HESSIAN_CONVENTION: analytic H*v disagrees with gradient differences.")

        k_bohr = float(np.einsum("ax,abxy,by->", tangent, hessian, tangent))
        mu_amu = float(np.einsum("a,ax,ax->", masses, tangent, tangent))
        summary.update(k_q_Eh_per_Bohr2=k_bohr, k_q_Eh_per_A2=k_bohr / BOHR_A**2,
                       k_q_N_per_m=k_bohr * EH_J / BOHR_M**2, mu_eff_amu=mu_amu, mu_eff_kg=mu_amu * AMU_KG)
        if name in DIATOMICS:
            reduced_mass = float(np.prod(masses) / np.sum(masses))
            error = abs(mu_amu / reduced_mass - 1)
            axis = (reference[1] - reference[0]) / path.s_eq
            ordinary_k = float(axis @ hessian[1, 1] @ axis)
            result["diatomic_checks"] = {
                "ordinary_reduced_mass_amu": reduced_mass, "mass_relative_error": error,
                "ordinary_bond_curvature_Eh_per_Bohr2": ordinary_k,
                "curvature_relative_error": abs(ordinary_k / k_bohr - 1) if k_bohr else None,
            }
            if error > TOL["diatomic_mass_relative"] or abs(ordinary_k - k_bohr) > 1e-7:
                result["failures"].append("FAIL_DIATOMIC: generalized mass or curvature disagrees with ordinary bond stretch.")

        acceleration_step = TANGENT_STEPS[-1]
        acceleration = (path(acceleration_step) - 2 * reference + path(-acceleration_step)) / acceleration_step**2
        gradient_correction = float(np.sum(gradient * acceleration) * BOHR_A)
        result["path_acceleration"] = {"R_second_derivative_per_A": acceleration,
                                       "gradient_correction_Eh_per_Bohr2": gradient_correction}
        fd_rows = []
        for h in ENERGY_STEPS:
            energies = [run_rhf(symbols, path(sign * h), basis, mf.make_rdm1()).e_tot for sign in (1, -1)]
            second = math.fsum([float(energies[0]), float(energies[1]), -2 * float(mf.e_tot)])
            fd_a = second / h**2
            fd_bohr = second / (h / BOHR_A)**2
            fd_rows.append({"h_A": h, "E_plus_Eh": float(energies[0]), "E_zero_Eh": float(mf.e_tot),
                            "E_minus_Eh": float(energies[1]), "k_FD_Eh_per_A2": fd_a,
                            "k_FD_Eh_per_Bohr2": fd_bohr,
                            "relative_difference": abs(fd_bohr - k_bohr) / max(abs(k_bohr), 1e-30),
                            "difference_from_acceleration_corrected_H_Eh_per_Bohr2": fd_bohr - k_bohr - gradient_correction})

        fine, coarse = fd_rows[-1], fd_rows[-2]
        richardson = (4 * fine["k_FD_Eh_per_Bohr2"] - coarse["k_FD_Eh_per_Bohr2"]) / 3
        allowed_fd = TOL["fd_absolute_Eh_per_Bohr2"] + TOL["fd_relative"] * abs(k_bohr)
        fd_ok = all(abs(r["k_FD_Eh_per_Bohr2"] - k_bohr) <= allowed_fd for r in (coarse, fine))
        fd_ok = fd_ok and abs(fine["k_FD_Eh_per_Bohr2"] - coarse["k_FD_Eh_per_Bohr2"]) <= allowed_fd
        result["finite_differences"] = {
            "rows": fd_rows, "comparison_h_A": fine["h_A"], "richardson_Eh_per_Bohr2": richardson,
            "richardson_relative_difference": abs(richardson - k_bohr) / max(abs(k_bohr), 1e-30),
            "acceptance_absolute_tolerance_Eh_per_Bohr2": allowed_fd,
            "convergence_pass": fd_ok,
        }
        summary["Hessian_FD_relative_difference"] = fine["relative_difference"]
        if not fd_ok:
            result["failures"].append("FAIL_FD: finest two energy curvatures disagree with H or each other; inspect gradient correction and convergence table.")
        if k_bohr <= 0 or mu_amu <= 0:
            result["failures"].append("FAIL_CURVATURE_OR_MASS: positive curvature and mass are required.")
        else:
            ranges, oscillator = oscillator_ranges(k_bohr, mu_amu, path.s_eq)
            summary.update(ranges)
            result["harmonic_oscillator"] = oscillator
            modes = normal_modes(mol, hessian, tangent, masses, k_bohr, mu_amu)
            result["normal_mode_diagnostic"] = modes
            for key in ("dominant_normal_mode", "dominant_mode_frequency_cm1", "mode_overlap"):
                summary[key] = modes[key]
            if modes["mass_orthogonality_max_error"] > TOL["mode_orthogonality"] or abs(modes["sum_squared_weights"] - 1) > TOL["mode_weight_sum"] or modes["curvature_reconstruction_relative_error"] > TOL["mode_curvature_relative"]:
                result["failures"].append("FAIL_MODE_CONVENTION: mass normalization or spectral curvature reconstruction failed.")
            if modes["imaginary_mode_count"]:
                result["failures"].append("FAIL_EQUILIBRIUM_MINIMUM: imaginary vibrational modes were found.")
            if not np.isfinite(list(ranges.values())).all() or ranges["n1_min_A"] <= 0:
                result["failures"].append("FAIL_RANGE: nonfinite extents or nonpositive bond interval endpoint.")
        summary["validation_status"] = "PASS" if not result["failures"] else "FAIL_VALIDATION"
    except InputProblem as exc:
        summary["validation_status"] = "NEEDS_INPUT"
        result["failures"].append(str(exc))
    except Exception as exc:
        summary["validation_status"] = "ERROR"
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    finally:
        summary["message"] = "; ".join(result["failures"])
        result["elapsed_seconds"] = time.monotonic() - start
    return result


def validate_scan_geometry(path, q, coords):
    original, current = path.internals(path.ref), path.internals(coords)
    length_tol = max(5e-10, 256 * np.finfo(float).eps * max(1, float(np.max(np.abs(coords)))))
    fixed_lengths = [v["value"] for v in original.values() if v["kind"] == "fixed_bond"]
    angle_tol = max(2e-7, math.degrees(8 * length_tol / min(fixed_lengths or [1.])))
    errors, passed = {}, bool(np.isfinite(coords).all())
    for label, item in original.items():
        expected = item["value"] + q if item["kind"] == "stretch" else item["value"]
        difference = (angular_difference(current[label]["value"], expected)
                      if item["kind"] == "fixed_dihedral" else current[label]["value"] - expected)
        errors[label] = abs(difference)
        passed = passed and abs(difference) <= (length_tol if item["unit"] == "A" else angle_tol)
    return {"passed": passed, "absolute_internal_errors": errors,
            "length_tolerance_A": length_tol, "angle_tolerance_deg": angle_tol}


def orbital_stability(mf):
    gradient, operator, _ = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ, with_symmetry=False)
    dimension = gradient.size
    matrix = np.column_stack([2 * operator(v) for v in np.eye(dimension)])
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return gradient * 2, matrix, eigenvalues, eigenvectors


def configured_rhf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-13, 1e-12, 1e-14
    mf.max_cycle, mf.diis_space, mf.conv_check = 400, 12, False
    mf.direct_scf = False
    mf.chkfile = None
    return mf


def polish_orbitals(mf):
    history, previous_energy = [], None
    for iteration in range(30):
        energy = float(mf.energy_tot(dm=mf.make_rdm1()))
        gradient, matrix, eigenvalues, vectors = orbital_stability(mf)
        residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
        delta = None if previous_energy is None else abs(energy - previous_energy)
        history.append({"iteration": iteration, "energy_Eh": energy, "orbital_gradient_norm_Eh": residual,
                        "energy_change_Eh": delta, "lowest_internal_eigenvalue_Eh": float(eigenvalues[0])})
        if residual <= 1e-11 and delta is not None and delta <= 1e-13:
            mf.e_tot, mf.converged = energy, True
            return True, history
        previous_energy = energy
        if residual <= 1e-11:
            continue
        keep = abs(eigenvalues) > 2e-10
        step = -vectors[:, keep] @ ((vectors[:, keep].T @ gradient) / eigenvalues[keep])
        step *= min(1., .1 / max(float(np.linalg.norm(step)), 1e-30))
        if not np.isfinite(step).all():
            break
        mf.mo_coeff = mf.mo_coeff @ expm(scf.hf.unpack_uniq_var(step, mf.mo_occ))
    mf.e_tot = float(mf.energy_tot(dm=mf.make_rdm1()))
    mf.converged = False
    return False, history


def newton_restart(mf, orbitals=None):
    solver = mf.newton()
    solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-13, 1e-12, 400
    solver.ah_conv_tol, solver.ah_lindep = 1e-14, 1e-20
    solver.kernel(mo_coeff=mf.mo_coeff if orbitals is None else orbitals, mo_occ=mf.mo_occ)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = solver.mo_coeff, solver.mo_occ, solver.mo_energy
    mf.e_tot, mf.converged = solver.e_tot, solver.converged
    return bool(solver.converged)


def solve_point(symbols, coords, basis, density=None, independent=False, previous=None):
    attempts = []
    guesses = [("fresh_minao" if density is None else "supplied_density", density)]
    if previous is not None:
        guesses.insert(0, ("orthonormalized_orbital_continuation", None))
    if density is not None and not independent:
        guesses.append(("fresh_minao_fallback", None))
    last_state = {"accepted": False, "scf_converged": False, "energy_Eh": None,
                  "orbital_gradient_norm_Eh": None, "internal_stable": False,
                  "restricted_stability_corrections": 0, "solver_attempts": attempts}
    for label, guess in guesses:
        attempt = {"initial_guess": label, "stability_repairs": []}
        attempts.append(attempt)
        try:
            mf = configured_rhf(make_molecule(symbols, coords, basis))
            if label == "orthonormalized_orbital_continuation":
                metric = previous.mo_coeff.T @ mf.get_ovlp() @ previous.mo_coeff
                metric_values, metric_vectors = np.linalg.eigh(metric)
                if np.min(metric_values) <= 1e-10:
                    raise ValueError("Transported orbital overlap is singular.")
                inverse_sqrt = (metric_vectors / np.sqrt(metric_values)) @ metric_vectors.T
                mf.mo_coeff = previous.mo_coeff @ inverse_sqrt
                mf.mo_occ, mf.mo_energy = previous.mo_occ.copy(), previous.mo_energy.copy()
                attempt["initial_solver"] = "transported paired orbitals; full-Fock Newton refinement"
            else:
                mf.kernel(dm0=guess)
                attempt["initial_pyscf_converged"] = bool(mf.converged)
            good, history = polish_orbitals(mf)
            attempt["initial_polish"] = history
            if not good:
                attempt["newton_restart_converged"] = newton_restart(mf)
                good, history = polish_orbitals(mf)
                attempt["restart_polish"] = history
            stable, eigenvalues = False, np.array([float("nan")])
            for repair in range(9):
                if not good:
                    break
                _, _, eigenvalues, vectors = orbital_stability(mf)
                stable = bool(eigenvalues[0] >= STABILITY_LIMIT)
                if stable or repair == 8:
                    break
                energy_before = float(mf.e_tot)
                alternatives = []
                for sign in (-1., 1.):
                    orbitals = mf.mo_coeff @ expm(scf.hf.unpack_uniq_var(sign * .35 * vectors[:, 0], mf.mo_occ))
                    trial_energy = float(mf.energy_tot(dm=mf.make_rdm1(orbitals, mf.mo_occ)))
                    alternatives.append((trial_energy, orbitals))
                trial_energy, orbitals = min(alternatives, key=lambda entry: entry[0])
                newton_converged = newton_restart(mf, orbitals)
                good, history = polish_orbitals(mf)
                attempt["stability_repairs"].append({
                    "energy_before_Eh": energy_before, "lowest_eigenvalue_before_Eh": float(eigenvalues[0]),
                    "initial_rotation_energy_Eh": trial_energy, "energy_after_Eh": float(mf.e_tot),
                    "newton_converged": newton_converged, "polish": history,
                })
            residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
            if not good:
                _, _, eigenvalues, _ = orbital_stability(mf)
                stable = bool(eigenvalues[0] >= STABILITY_LIMIT)
            electron_trace = float(np.einsum("ij,ji->", mf.make_rdm1(), mf.get_ovlp()))
            orthogonality = float(np.max(np.abs(mf.mo_coeff.T @ mf.get_ovlp() @ mf.mo_coeff - np.eye(mf.mo_coeff.shape[1]))))
            accepted = bool(good and residual <= SCF_GRADIENT_LIMIT and stable and np.isfinite(mf.e_tot)
                            and abs(electron_trace - mf.mol.nelectron) < 1e-8 and orthogonality < 1e-9
                            and np.all(np.isin(mf.mo_occ, [0, 2])) and getattr(mf, "with_df", None) is None)
            state = {"accepted": accepted, "scf_converged": bool(good), "energy_Eh": float(mf.e_tot),
                     "orbital_gradient_norm_Eh": residual, "internal_stable": stable,
                     "lowest_internal_stability_eigenvalue_Eh": float(eigenvalues[0]),
                     "restricted_stability_corrections": len(attempt["stability_repairs"]),
                     "electron_number": electron_trace, "orbital_orthogonality_max_error": orthogonality,
                     "solver_attempts": attempts}
            attempt["final_energy_Eh"], attempt["accepted"] = float(mf.e_tot), accepted
            last_state = state
            if accepted:
                _, charges = mf.mulliken_pop(verbose=0)
                state["mulliken_atomic_charges"] = charges
                return mf, state
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
    last_state["failure"] = "RHF failed strict energy/gradient, internal-stability or orbital consistency checks."
    return None, last_state


def starting_geometries(name):
    if name in DIATOMICS:
        symbols = {"N2": ["N", "N"], "CO": ["C", "O"], "HF": ["H", "F"]}[name]
        length = {"N2": 1.2, "CO": 1.25, "HF": 1.05}[name]
        return [(symbols, np.array([[0., 0., 0.], [0., 0., length]]))]
    if name == "H2O":
        x, z = .8, .65
        return [(["O", "H", "H"], np.array([[0., 0., 0.], [x, 0., z], [-x, 0., z]]))]
    if name == "NH3":
        coords = [[0., 0., 0.]] + [[math.cos(phi), math.sin(phi), .4] for phi in (0., 2*math.pi/3, 4*math.pi/3)]
        return [(["N", "H", "H", "H"], np.array(coords))]
    if name == "H2O2":
        return [(["O", "O", "H", "H"], np.array([[0., 0., -.75], [0., 0., .75],
                  [1., 0., -1.], [math.cos(phi), math.sin(phi), 1.]]))
                for phi in np.deg2rad([60., 120., 160.])]
    raise ValueError(f"Unsupported molecule: {name}")


def optimize_equilibrium(name, basis, geometry_dir, gtol=1e-6, maxiter=200, starts=None):
    candidates, errors = [], []
    for index, (symbols, initial) in enumerate(starts or starting_geometries(name), 1):
        history = []
        def objective(flat):
            coords_A = flat.reshape(-1, 3) * BOHR_A
            mf = run_rhf(symbols, coords_A, basis)
            gradient = mf.nuc_grad_method().kernel()
            if not np.isfinite(gradient).all():
                raise ValueError("Nonfinite RHF nuclear gradient during optimization.")
            history.append({"E_RHF_Eh": float(mf.e_tot),
                            "cartesian_A": coords_A.copy(),
                            "gradient_Eh_per_Bohr": gradient.copy(),
                            "max_gradient_Eh_per_Bohr": float(np.max(np.abs(gradient)))})
            return float(mf.e_tot), gradient.ravel()
        try:
            initial = np.array(initial, float)
            initial -= initial.mean(axis=0)
            optimized = minimize(objective, (initial / BOHR_A).ravel(), method="BFGS", jac=True,
                                 options={"gtol": gtol, "maxiter": maxiter})
            coords_A = optimized.x.reshape(-1, 3) * BOHR_A

            energy, gradient = objective(optimized.x)
            max_gradient = float(np.max(np.abs(gradient)))
            passed = bool(optimized.success and max_gradient <= gtol)
            candidates.append({"start": index, "symbols": symbols, "cartesian_A": coords_A,
                               "E_eq_Eh": energy, "max_gradient_Eh_per_Bohr": max_gradient,
                               "optimizer_success": bool(optimized.success), "iterations": int(optimized.nit),
                               "evaluations": history, "converged": passed, "message": str(optimized.message)})
        except Exception as exc:
            errors.append(f"Start {index}: {type(exc).__name__}: {exc}")
    converged = [c for c in candidates if c["converged"]]
    if not converged:
        return {"status": "FAIL_RHF_EQUILIBRIUM", "candidates": candidates, "errors": errors,
                "message": "No starting geometry reached a converged RHF equilibrium."}
    best = min(converged, key=lambda c: c["E_eq_Eh"])
    geometry_dir.mkdir(parents=True, exist_ok=True)
    xyz = geometry_dir / f"{name}_converged.xyz"
    lines = [str(len(best["symbols"])), f"{name} RHF/{basis} converged E={best['E_eq_Eh']:.15f} Eh"]
    lines += [f"{symbol} {r[0]:.14f} {r[1]:.14f} {r[2]:.14f}" for symbol, r in zip(best["symbols"], best["cartesian_A"])]
    xyz.write_text("\n".join(lines) + "\n")
    row = {"molecule": name, "method": "RHF", "basis": basis, "status": "converged",
           "optimizer_success": "True", "energy_hartree": best["E_eq_Eh"], "xyz_file": str(xyz.resolve()),
           "max_gradient_hartree_per_bohr": best["max_gradient_Eh_per_Bohr"], "iterations": best["iterations"]}
    metadata = geometry_dir / f"{name}_equilibrium.csv"
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return {"status": "PASS", "row": row, "metadata_path": str(metadata), "source": "computed",
            "symbols": best["symbols"], "cartesian_A": best["cartesian_A"], "E_eq_Eh": best["E_eq_Eh"],
            "max_gradient_Eh_per_Bohr": best["max_gradient_Eh_per_Bohr"],
            "selected_start": best["start"], "candidates": candidates, "errors": errors}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def n1_summary(name, equilibrium, harmonic):
    hs = harmonic["summary"]
    require(hs["validation_status"] == "PASS", "Harmonic validation must pass.")
    energy = float(equilibrium["E_eq_Eh"])
    require(abs(energy - harmonic["stationarity"]["E_RHF_Eh"]) < 1e-9,
            "Equilibrium and harmonic RHF energies disagree.")
    quantum = HBAR * hs["omega_rad_per_s"] / EH_J
    excitation = 1.5 * quantum
    require(0 < hs["n1_min_A"] < hs["n1_max_A"], "Invalid n=1 interval.")
    return {
        "molecule": name, "basis": "sto-3g", "coordinate": hs["coordinate_definition"],
        "r_e_A": hs["s_eq_A"], "E_RHF_re_Ha": energy,
        "k_Ha_per_Bohr2": hs["k_q_Eh_per_Bohr2"], "k_Ha_per_A2": hs["k_q_Eh_per_A2"],
        "k_N_per_m": hs["k_q_N_per_m"], "mu_eff_amu": hs["mu_eff_amu"],
        "omega_rad_per_s": hs["omega_rad_per_s"], "hbar_omega_Ha": quantum,
        "E_n0_Ha": 0.5 * quantum, "E_n1_Ha": excitation,
        "E_total_n1_Ha": energy + excitation, "n1_total_is_negative": energy + excitation < 0,
        "n": 1, "Delta_q_n1_A": hs["Delta_q_n1_A"],
        "n1_min_A": hs["n1_min_A"], "n1_max_A": hs["n1_max_A"],
        "range_formula": "r_e +/- sqrt(3*hbar/(mu_eff*omega)); E_n1 = 1.5*hbar*omega",
        "energy_reference": "Absolute electronic totals include nuclear repulsion; no energy shift.",
        "zero_test_note": "The sign of E_RHF(re)+E_n1 is recorded only; it is not a physical binding criterion.",
    }


def scan_grid(summary, count):
    require(type(count) is int and count >= 2, "At least two scan points are required.")
    lower, upper = summary["n1_min_A"], summary["n1_max_A"]
    require(math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper,
            "Invalid scan endpoints.")
    grid = np.linspace(lower, upper, count)
    grid[0], grid[-1] = lower, upper
    return grid


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_xyz(path, name, symbols, coords, label):
    lines = [str(len(symbols)), f"{name} {label}; units=Angstrom"]
    lines += [f"{symbol} {x:.15f} {y:.15f} {z:.15f}"
              for symbol, (x, y, z) in zip(symbols, coords)]
    atomic_text(path, "\n".join(lines) + "\n")


def checkpoint_reference(path, output_dir):
    return {"path": str(path.relative_to(output_dir)), "sha256": sha256(path)}


def rhf_arrays(mf):
    return {
        "mo_coeff": mf.mo_coeff, "mo_occ": mf.mo_occ, "mo_energy_Ha": mf.mo_energy,
        "rhf_density_ao": mf.make_rdm1(), "overlap_ao": mf.get_ovlp(),
        "hcore_ao_Ha": mf.get_hcore(), "fock_ao_Ha": mf.get_fock(),
        "molecule_pyscf_json": np.asarray(mf.mol.dumps()), "E_RHF_Ha": np.asarray(mf.e_tot),
    }


def frozen_core_fci(mf, name, checkpoint=None):
    ncore, ncas, nelec = ACTIVE_SPACES[name]
    require(mf.mol.nelectron == nelec + 2 * ncore and mf.mo_coeff.shape[1] == ncas + ncore,
            "Orbital/electron counts do not match the frozen-core STO-3G convention.")
    mf.mo_energy, mf.mo_coeff = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    require(np.all(np.isin(mf.mo_occ, [0, 2])) and np.all(np.diff(mf.mo_occ) <= 0)
            and np.all(mf.mo_occ[:ncore] == 2), "Invalid closed-shell RHF occupations.")
    require(np.all(np.diff(mf.mo_energy[mf.mo_occ == 2]) >= -1e-10),
            "Occupied canonical orbitals are not energy ordered.")
    mc = mcscf.CASCI(mf, ncas, nelec, ncore=ncore)
    mc.fcisolver.conv_tol, mc.fcisolver.max_cycle = 1e-12, 200
    mc.verbose = 0
    try:
        total = float(mc.kernel()[0])
    except Exception as exc:
        if checkpoint:
            checkpoint({"fci_converged": False, "fci_error": f"{type(exc).__name__}: {exc}"}, {})
        raise
    h1, core = mc.get_h1eff()
    active = float(mc.e_cas)
    record = {
        "electronic_state_stage": "FCI_ATTEMPT",
        "E_FCI_Ha": total, "E_FCI_frozen_core_Ha": total,
        "E_CAS_active_Ha": active, "E_core_including_nuclear_Ha": float(core),
        "E_nuclear_Ha": float(mf.mol.energy_nuc()),
        "n_frozen_orbitals": ncore, "n_frozen_electrons": 2 * ncore,
        "n_active_orbitals": ncas, "n_active_electrons": nelec,
        "canonical_rhf_orbitals": True, "fci_converged": bool(mc.fcisolver.converged),
        "fci_energy_tolerance_Ha": mc.fcisolver.conv_tol, "fci_max_cycle": mc.fcisolver.max_cycle,
        "fci_convention": "frozen-core", "total_energy_includes_nuclear_repulsion": True,
        "energy_decomposition": "E_FCI_Ha = E_CAS_active_Ha + E_core_including_nuclear_Ha",
    }
    arrays = {
        **rhf_arrays(mf),
        "ci_vector": np.asarray(mc.ci), "active_h1_Ha": h1,
        "active_h2_packed_Ha": mc.get_h2eff(), "core_energy_Ha": np.asarray(core),
        "E_FCI_Ha": np.asarray(total),
        "E_corr_Ha": np.asarray(total - mf.e_tot),
    }

    if checkpoint:
        checkpoint(record, arrays)
    require(bool(mc.fcisolver.converged), "Frozen-core FCI did not converge.")
    require(np.isfinite([total, active, core]).all(), "Nonfinite FCI energy.")
    require(abs(total - active - core) < 1e-9, "FCI total does not equal active plus core energy.")
    require(total <= mf.e_tot + 1e-9, "FCI energy exceeds the matching RHF reference.")
    return record, arrays


def calculate_point(name, symbols, coords, bond_length, q, index, folder, output_dir, geometry_check=None):
    folder.mkdir(parents=True, exist_ok=True)
    point = {
        "molecule": name, "point_index": index, "basis": "sto-3g", "charge": 0, "spin": 0,
        "symbols": list(symbols), "cartesian_A": jsonable(coords),
        "bond_length_A": float(bond_length), "q_A": float(q), "status": "RUNNING",
    }
    if geometry_check is not None:
        point["geometry_validation"] = geometry_check
    start = time.monotonic()
    arrays = {"cartesian_A": np.asarray(coords), "symbols": np.asarray(symbols)}

    def electronic_checkpoint(record, new_arrays):
        point.update(record)
        arrays.update(new_arrays)
        archive = folder / "electronic_state.npz"
        with archive.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        archive.with_suffix(".npz.tmp").replace(archive)
        point["electronic_state_file"] = checkpoint_reference(archive, output_dir)
        atomic_text(folder / "point.json", json.dumps(jsonable(point), indent=2, allow_nan=False) + "\n")

    try:
        xyz = folder / "geometry.xyz"
        write_xyz(xyz, name, symbols, coords, f"point={index}; q={q:.15g}")
        point["geometry_file"] = checkpoint_reference(xyz, output_dir)
        atomic_text(folder / "point.json", json.dumps(jsonable(point), indent=2, allow_nan=False) + "\n")
        mf, diagnostics = solve_point(symbols, np.asarray(coords), "sto-3g")
        point["rhf_diagnostics"] = diagnostics
        require(mf is not None and diagnostics["accepted"], "RHF point failed validation.")
        point["E_RHF_Ha"] = float(mf.e_tot)
        electronic_checkpoint({"electronic_state_stage": "RHF"}, rhf_arrays(mf))
        fci_record, _ = frozen_core_fci(mf, name, checkpoint=electronic_checkpoint)
        point.update(fci_record)
        point["E_corr_Ha"] = point["E_FCI_Ha"] - point["E_RHF_Ha"]
        point["E_corr_mHa"] = 1000 * point["E_corr_Ha"]
        point["electronic_state_stage"] = "RHF_AND_FCI"
        point["status"] = "PASS"
    except Exception as exc:
        point["status"] = "FAIL"
        point["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        point["elapsed_seconds"] = time.monotonic() - start
        atomic_text(folder / "point.json", json.dumps(jsonable(point), indent=2, allow_nan=False) + "\n")
    return jsonable(point)

SCAN_FIELDS = (
    "molecule", "point_index", "basis", "bond_length_A", "q_A", "E_RHF_Ha", "E_FCI_Ha",
    "E_corr_Ha", "E_corr_mHa", "E_CAS_active_Ha", "E_core_including_nuclear_Ha",
    "E_nuclear_Ha", "n_frozen_orbitals", "n_active_orbitals", "n_active_electrons", "status", "error",
)
SUMMARY_FIELDS = (
    "molecule", "basis", "r_e_A", "E_RHF_re_Ha", "E_FCI_re_Ha", "E_corr_re_Ha",
    "k_Ha_per_Bohr2", "k_Ha_per_A2", "k_N_per_m", "mu_eff_amu", "omega_rad_per_s",
    "hbar_omega_Ha", "E_n0_Ha", "E_n1_Ha", "E_total_n1_Ha", "n1_total_is_negative",
    "Delta_q_n1_A", "n1_min_A", "n1_max_A", "scan_points", "completed_points", "status", "error",
)


def csv_text(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def save_outputs(document, output_dir):
    document["updated_utc"] = datetime.now(timezone.utc).isoformat()
    summaries, scans, equilibria = [], [], []
    for result in document["results"]:
        summary = dict(result.get("summary", {}))
        reference = result.get("equilibrium_energies", {})
        summary.update(molecule=result["molecule"], status=result["status"], error=result.get("error", ""),
                       scan_points=document["settings"]["scan_points"],
                       completed_points=sum(p.get("status") == "PASS" for p in result.get("scan", [])),
                       E_FCI_re_Ha=reference.get("E_FCI_Ha"), E_corr_re_Ha=reference.get("E_corr_Ha"))
        summaries.append(summary)
        scans.extend(result.get("scan", []))
        if reference:
            equilibria.append(reference)
        atomic_text(output_dir / result["molecule"] / "result.json",
                    json.dumps(jsonable(result), indent=2, allow_nan=False) + "\n")
    atomic_text(output_dir / "summary.csv", csv_text(summaries, SUMMARY_FIELDS))
    atomic_text(output_dir / "scan.csv", csv_text(scans, SCAN_FIELDS))
    atomic_text(output_dir / "equilibrium_energies.csv", csv_text(equilibria, SCAN_FIELDS))

    atomic_text(output_dir / "initialdata.json", json.dumps(jsonable(document), indent=2, allow_nan=False) + "\n")


def validate_saved_point(point, name, symbols, coords, index, length, q, output_dir):
    require(point["status"] == "PASS" and point["molecule"] == name and point["point_index"] == index,
            "Saved point identity/status mismatch.")
    require(point["basis"] == "sto-3g" and point["symbols"] == list(symbols), "Saved point basis/atoms mismatch.")
    require(np.asarray(point["cartesian_A"]).shape == np.asarray(coords).shape
            and np.allclose(point["cartesian_A"], coords, atol=1e-12, rtol=0), "Saved geometry changed.")
    require(abs(point["bond_length_A"] - length) < 1e-12 and abs(point["q_A"] - q) < 1e-12,
            "Saved scan coordinate changed.")
    require(point["rhf_diagnostics"]["accepted"] and point["fci_converged"], "Unvalidated cached solver result.")
    values = [point[k] for k in ("E_RHF_Ha", "E_FCI_Ha", "E_corr_Ha", "E_corr_mHa",
                                "E_CAS_active_Ha", "E_core_including_nuclear_Ha")]
    require(np.isfinite(values).all(), "Nonfinite saved energy.")
    require(abs(point["E_FCI_Ha"] - point["E_RHF_Ha"] - point["E_corr_Ha"]) < 1e-12,
            "Saved correlation energy is inconsistent.")
    require(abs(point["E_corr_mHa"] - 1000 * point["E_corr_Ha"]) < 1e-10, "Saved correlation units differ.")
    require(abs(point["E_FCI_Ha"] - point["E_CAS_active_Ha"] - point["E_core_including_nuclear_Ha"]) < 1e-9,
            "Saved FCI decomposition differs.")
    for key in ("geometry_file", "electronic_state_file"):
        reference = point[key]
        path = (output_dir / reference["path"]).resolve()
        require(path.is_relative_to(output_dir.resolve()), "Checkpoint path lies outside the output folder.")
        require(path.is_file() and sha256(path) == reference["sha256"], f"Missing or changed {key}.")


def recover_point(record, path):
    if record and record.get("status") == "PASS":
        return record
    if path.is_file():
        candidate = json.loads(path.read_text())
        if candidate.get("status") == "PASS":
            return candidate
    return record


def process_molecule(result, args, document):
    name, output_dir = result["molecule"], args.output_dir
    result.update(status="RUNNING", error="")
    checkpoint = lambda: save_outputs(document, output_dir)
    folder = output_dir / name
    try:
        if result.get("equilibrium", {}).get("status") != "PASS":
            print(f"{name}: optimizing RHF equilibrium", flush=True)
            result["equilibrium"] = optimize_equilibrium(name, "sto-3g", folder / "equilibrium",
                                                        gtol=args.gtol, maxiter=args.maxiter)
            checkpoint()
        equilibrium = result["equilibrium"]
        require(equilibrium["status"] == "PASS", "RHF equilibrium optimization failed.")
        if result.get("harmonic", {}).get("summary", {}).get("validation_status") != "PASS":
            print(f"{name}: calculating and validating k and the n=1 range", flush=True)
            result["harmonic"] = calculate_harmonic(name, equilibrium["row"], Path(equilibrium["metadata_path"]))
            checkpoint()
        harmonic = result["harmonic"]
        require(harmonic["summary"]["validation_status"] == "PASS",
                "Harmonic calculation failed: " + "; ".join(harmonic.get("failures", [])))
        summary = n1_summary(name, equilibrium, harmonic)
        result["summary"] = summary
        symbols = harmonic["symbols"]
        path = StretchPath(name, symbols, harmonic["equilibrium_cartesian_A"],
                           harmonic["isotope_average_masses_amu"])
        tangent, validation = path.validate()
        require(not validation["failures"] and np.allclose(tangent, harmonic["tangent_dR_dq"], atol=2e-8, rtol=0),
                "The scan path does not reproduce the harmonic tangent.")
        reference = recover_point(result.get("equilibrium_energies"), folder / "equilibrium_energies/point.json")
        if reference and reference.get("status") == "PASS":
            validate_saved_point(reference, name, symbols, path.ref, 0, path.s_eq, 0., output_dir)
            result["equilibrium_energies"] = reference
        else:
            print(f"{name}: calculating FCI at the RHF equilibrium", flush=True)
            reference = calculate_point(name, symbols, path.ref, path.s_eq, 0., 0,
                                        folder / "equilibrium_energies", output_dir)
            result["equilibrium_energies"] = reference
            checkpoint()
        require(reference["status"] == "PASS", "Equilibrium RHF/FCI failed: " + reference.get("error", ""))
        require(abs(reference["E_RHF_Ha"] - summary["E_RHF_re_Ha"]) < 1e-9,
                "Stable RHF equilibrium energy differs from the optimization/harmonic reference.")
        grid = scan_grid(summary, document["settings"]["scan_points"])
        points = result.setdefault("scan", [])
        require(len({p["point_index"] for p in points}) == len(points), "Duplicate saved scan indices.")
        require(all(type(p["point_index"]) is int and 1 <= p["point_index"] <= len(grid) for p in points),
                "Unexpected saved scan indices.")
        for index, length in enumerate(grid, 1):
            q = float(length - path.s_eq)
            coords = path(q)
            geometry_check = validate_scan_geometry(path, q, coords)
            require(geometry_check["passed"], f"Invalid scan geometry at point {index}.")
            cached = next((p for p in points if p["point_index"] == index), None)
            recovered = recover_point(cached, folder / "scan" / f"{index:03d}" / "point.json")
            if recovered is not cached:
                validate_saved_point(recovered, name, symbols, coords, index, length, q, output_dir)
                if cached is None:
                    points.append(recovered)
                else:
                    points[points.index(cached)] = recovered
                points.sort(key=lambda p: p["point_index"])
                cached = recovered
            if cached and cached.get("status") == "PASS":
                validate_saved_point(cached, name, symbols, coords, index, length, q, output_dir)
                continue
            print(f"{name}: RHF/FCI scan {index}/{len(grid)} at {length:.9f} Angstrom", flush=True)
            point = calculate_point(name, symbols, coords, length, q, index,
                                    folder / "scan" / f"{index:03d}", output_dir, geometry_check)
            if cached is not None:
                points[points.index(cached)] = point
            else:
                points.append(point)
            points.sort(key=lambda p: p["point_index"])
            checkpoint()
        failed = [p["point_index"] for p in points if p["status"] != "PASS"]
        require(not failed and len(points) == len(grid), f"Incomplete or failed scan points: {failed}")
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"{name}: {result['error']}", file=sys.stderr, flush=True)
    finally:
        checkpoint()
    return result["status"] == "PASS"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", nargs="+", choices=[*MOLECULES, "all"], default=list(MOLECULES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INITIAL_DIR,
                        help="Output directory, relative to the project root unless absolute.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gtol", type=float, default=1e-6, help="RHF nuclear gradient tolerance in Ha/Bohr.")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--resume", action="store_true", help="Resume a matching run, or create one if absent.")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.maxiter < 1 or not math.isfinite(args.gtol) or not 0 < args.gtol <= 1e-6:
        parser.error("threads/maxiter must be positive and 0 < gtol <= 1e-6.")
    args.molecules = list(MOLECULES) if args.molecules == ["all"] else args.molecules
    if "all" in args.molecules:
        parser.error("Use all alone or select explicit molecule IDs")
    if len(set(args.molecules)) != len(args.molecules):
        parser.error("Molecules must not be repeated.")
    output_dir = args.output_dir.expanduser()
    args.output_dir = (output_dir if output_dir.is_absolute() else PROJECT / output_dir).resolve()
    return args


def run(args):
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "pyscf": pyscf.__version__}
    settings = {"basis": "sto-3g", "charge": 0, "spin": 0, "fci_convention": "frozen-core",
                "molecules": args.molecules, "scan_points": 30,
                "n": 1, "threads": args.threads, "gtol": args.gtol, "maxiter": args.maxiter,
                "active_spaces": jsonable(ACTIVE_SPACES)}
    manifest = args.output_dir / "initialdata.json"
    if manifest.exists():
        require(args.resume, "Output already exists. Use --resume or a new --output-dir.")
        document = json.loads(manifest.read_text())
        require(document["schema_version"] == SCHEMA_VERSION and document["settings"] == settings,
                "Saved settings differ; use a new output directory.")
        require(document["provenance"]["script_sha256"] == sha256(__file__)
                and document["provenance"]["versions"] == versions,
                "Code or dependency versions changed; use a new output directory.")
        require([r["molecule"] for r in document["results"]] == args.molecules, "Saved molecule list differs.")
    else:
        require(not any(p.name != ".run.lock" for p in args.output_dir.iterdir()),
                "Output directory is not empty; use a new --output-dir.")
        document = {
            "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "settings": settings, "status": "RUNNING",
            "units": {"energy": "Hartree (Ha = Eh)", "length": "Angstrom", "mass": "amu",
                      "k": "Ha/Bohr^2 and Ha/Angstrom^2", "angular_frequency": "rad/s"},
            "provenance": {"script_sha256": sha256(__file__), "versions": versions,
                           "command": sys.argv, "source_snapshot": "source_initialdata.py",
                           "constants": {"Bohr_A": BOHR_A, "Hartree_J": EH_J, "hbar_J_s": HBAR,
                                         "amu_kg": AMU_KG, "c_m_per_s": C_MS}},
            "formulas": {"force_constant": "k = t^T H_RHF t; t = dR/dq",
                         "effective_mass": "mu_eff = sum_A m_A * |t_A|^2",
                         "frequency": "omega = sqrt(k_SI/mu_kg)",
                         "n1_energy": "E_n1 = 1.5*hbar*omega; E_total_n1 = E_RHF(re)+E_n1",
                         "n1_half_width": "Delta_q_n1 = sqrt(3*hbar/(mu_eff*omega))",
                         "correlation": "E_corr = E_FCI - E_RHF; E_corr_mHa = 1000*E_corr_Ha"},
            "results": [{"molecule": name, "status": "PENDING", "scan": []} for name in args.molecules],
        }
        atomic_text(args.output_dir / "source_initialdata.py", Path(__file__).read_text())
    lib.num_threads(args.threads)
    document["status"] = "RUNNING"
    save_outputs(document, args.output_dir)
    try:
        for result in document["results"]:
            process_molecule(result, args, document)
        document["status"] = "PASS" if all(r["status"] == "PASS" for r in document["results"]) else "FAIL"
    except KeyboardInterrupt:
        document["status"] = "INTERRUPTED"
        print("Interrupted; completed stages are saved. Use --resume to continue.", file=sys.stderr)
        return 130
    finally:
        save_outputs(document, args.output_dir)
    print(f"{document['status']}: results saved to {args.output_dir}", flush=True)
    return 0 if document["status"] == "PASS" else 1


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return run(args)
        except (ValueError, KeyError, OSError) as exc:
            print(f"initialdata: {exc}", file=sys.stderr)
            return 2

if __name__ == "__main__":
    raise SystemExit(main())
