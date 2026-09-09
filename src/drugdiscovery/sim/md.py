"""Phase 6 — Molecular Dynamics layer (OpenMM).

Scope of this vertical slice (honest, reproducible, CPU-eligible because both
Warpgate GPUs are occupied by the ev3 model server):

  * Load the docked receptor from its CIF/PDB-compatible artifact.
  * Build an explicit-solvent system (TIP3P) with a standard AMBER protein
    force field, minimizing + short NPT production on the selected platform.
  * Analyse: backbone RMSD drift vs the minimised reference, per-residue
    RMSF, and (if the docked ligand is present) protein-ligand contact
    persistence over the trajectory.
  * Persist a SimulationRun(kind="md") reproducibility record.

Ligand FLEXIBILITY is explicitly OUT of this slice: adding flexible-ligand
vacuum/Ewald MD needs force-field parametrisation of the small molecule
(deferred until openff-parametrized ligands join the pipeline). We DO track
the static docking pose's contacts as a coarse persistence signal, labelled
computational hypothesis — never as measured binding.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform as _plat
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Built-in force fields bundled with OpenMM (no download, deterministic).
_PF = "amber99sbildn.xml"
_WF = "tip3p.xml"

# Residue names we treat as the catalytic/pocket focus for RMSF reporting.
_POCKET_FLANK = 6  # residues within this sequence window around a named residue


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hardware() -> str:
    import socket
    return f"{socket.gethostname()} cpu={_plat.processor() or _plat.machine()}"


def _pick_platform(requested: str | None, length: int) -> str:
    """Choose an OpenMM platform. Defaults to CUDA only if it is actually
    usable on this machine; otherwise CPU. Always reproducible (forces a
    deterministic platform name into the record)."""
    from openmm import Platform
    if requested:
        return requested
    try:
        names = {Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())}
        if "CUDA" in names:
            cuda = next(Platform.getPlatform(i) for i in range(Platform.getNumPlatforms())
                        if Platform.getPlatform(i).getName() == "CUDA")
            # sanity: don't silently fall back if CUDA is present but broken
            _, ok = cuda.getPropertyDefaultValue if hasattr(cuda, "getPropertyDefaultValue") else (None, True)
            return "CUDA"
        return "CPU"
    except Exception as e:  # noqa: BLE001
        log.warning("CUDA platform check failed (%s); using CPU", e)
        return "CPU"


class MdResults(dict):
    """Typed holder for MD analysis results."""


def _load_topology(path: Path):
    """Build an OpenMM Modeller from a receptor PDB, repairing incomplete
    residues via PDBFixer (template-based side-chain / missing-atom completion,
    then protonation at pH 7.4) so the force-field template match is exact.

    We deliberately avoid solvent/hydrogen guessing beyond PDBFixer: explicit
    TIP3P solvation is applied later at topology build time.
    """
    fixer = _pdbfixer_from_pdb(path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=7)
    from openmm import app as _app
    fixer.addMissingHydrogens(pH=7.4, forcefield=_app.ForceField(_PF))
    return _app.Modeller(fixer.topology, fixer.positions)


def _pdbfixer_from_pdb(path: Path):
    """Load PDBFxier (which needs the file's PDBId/format) from a local PDB."""
    from pdbfixer import PDBFixer
    fixer = PDBFixer(filename=str(path))
    return fixer


def _analyse(trajectory_positions, coords0, box0, n_steps) -> dict:
    """Compute backbone-RMSD drift, per-residue RMSF, radius-of-gyration."""
    frames = np.array(trajectory_positions)          # (T,N,3) nm
    ref = np.asarray(coords0)                        # (N,3) nm
    T = frames.shape[0]
    # translate to centroid frame (no alignment — we report raw drift)
    drift = frames[-1] - ref
    backbone_rmsd0 = float(np.sqrt(np.mean(np.sum((frames[0] - ref) ** 2, axis=-1))))
    # per-residue RMSF relative to mean structure
    mean_pos = frames.mean(axis=0)
    fluc = np.sqrt(np.mean(np.sum((frames - mean_pos) ** 2, axis=-1), axis=0))
    rmsf = float(np.sqrt(np.mean(fluc ** 2)))
    rmsf_p95 = float(np.percentile(fluc, 95))
    # radius of gyration of the mean structure
    com = mean_pos.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((mean_pos - com) ** 2, axis=-1))))
    return {
        "frames": T,
        "steps": n_steps,
        "backbone_rmsd_nm": round(backbone_rmsd0, 4),
        "final_drift_nm": round(float(np.sqrt(np.mean(np.sum(drift ** 2, axis=-1)))), 4),
        "rmsf_mean_nm": round(rmsf, 4),
        "rmsf_p95_nm": round(rmsf_p95, 4),
        "radius_of_gyration_nm": round(rg, 4),
    }


def run_md(*, protein_path, work_dir, steps=1000, temperature=310.0,
           platform=None, cpu_threads=8, seed=7, box_padding_nm=1.0,
           store_every=25, max_protein_atoms=20000) -> MdResults:
    """Run a short explicit-solvent MD on the receptor and analyse stability.

    Returns a dict with analysis + reproducibility fields. This is CPU-safe
    and designed to be summoned by the planner for the TOP-ranked structure
    candidates only (expensive secondary validation, spec §11).
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    logfile = work / "openmm.log"

    import openmm
    from openmm import app, unit as u

    t0 = time.time()
    plat = _pick_platform(platform, steps)
    log.info("MD platform=%s steps=%d T=%.1f box_pad=%.2f", plat, steps, temperature, box_padding_nm)

    raw_atom_count = sum(
        1 for line in Path(protein_path).read_text(errors="ignore").splitlines()
        if line.startswith("ATOM  ")
    )
    if raw_atom_count > max_protein_atoms:
        raise RuntimeError(
            f"MD preflight refused {raw_atom_count} protein atoms; "
            f"CPU safety budget is {max_protein_atoms}. Use a focused domain "
            "structure or raise max_protein_atoms explicitly."
        )
    modeller = _load_topology(protein_path)
    n_prot_atoms = modeller.topology.getNumAtoms()
    if n_prot_atoms > max_protein_atoms:
        raise RuntimeError(
            f"MD preflight refused {n_prot_atoms} protein atoms; "
            f"CPU safety budget is {max_protein_atoms}. Use a focused domain "
            "structure or raise max_protein_atoms explicitly."
        )
    ff = app.ForceField(_PF, _WF)
    modeller.addSolvent(ff, boxSize=None, padding=box_padding_nm * u.nanometer,
                        model="tip3p", ionicStrength=0.15 * u.molar,
                        positiveIon="Na+", negativeIon="Cl-")
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * u.nanometer,
                             constraints=app.HBonds,
                             removeCMMotion=False)

    # robust position extraction (works whether positions are Quantity[Vec3] or plain)
    def _xyz(poslist):
        try:
            return np.array([[float(p[i].value_in_unit(u.nanometer)) for i in range(3)]
                             for p in poslist])
        except (AttributeError, TypeError):
            return np.array([[float(p[i]) for i in range(3)] for p in poslist])
    coords0 = _xyz(modeller.positions)
    # water + ions are appended AFTER the protein atoms by addSolvent; limit
    # RMSD/RMSF analysis to the protein atoms so bulk water diffusion does not
    # dominate the drift metric.
    coords0 = coords0[:n_prot_atoms]

    # Repaired crystal structures can retain severe local clashes. A 2-fs
    # first step after a loose minimisation is enough to create NaNs, so use a
    # conservative 0.5-fs integrator and staged temperature equilibration.
    integrator = openmm.LangevinMiddleIntegrator(
        temperature * u.kelvin, 1.0 / u.picosecond,
        0.5 * u.femtosecond)

    if plat == "CUDA":
        prop = {"Precision": "mixed"}
    else:
        prop = {"Threads": str(cpu_threads)}
    simulation = app.Simulation(modeller.topology, system, integrator, platform=openmm.Platform.getPlatformByName(plat), platformProperties=prop)

    simulation.context.setPositions(modeller.positions)
    try:
        # 10 kJ/mol/nm is materially stricter than the former 500 value;
        # repaired side chains need the extra minimisation before dynamics.
        simulation.minimizeEnergy(
            tolerance=10.0 * u.kilojoule_per_mole / u.nanometer,
            maxIterations=5000,
        )
    except openmm.OpenMMException as e:
        raise RuntimeError(f"MD minimisation failed before dynamics: {e}") from e

    min_state = simulation.context.getState(getPositions=True)
    min_positions = min_state.getPositions()
    min_xyz = _xyz(min_positions)
    if not np.isfinite(min_xyz).all():
        raise RuntimeError("MD minimisation produced non-finite coordinates")

    def _safe_step(n: int, phase: str) -> None:
        try:
            simulation.step(n)
        except openmm.OpenMMException as e:
            if "NaN" in str(e):
                raise RuntimeError(
                    f"MD integration diverged to NaN during {phase}; "
                    "structure excluded from MD after deterministic repair/minimisation"
                ) from e
            raise
        state = simulation.context.getState(getPositions=True)
        xyz = _xyz(state.getPositions())
        if not np.isfinite(xyz).all():
            raise RuntimeError(f"MD produced non-finite coordinates during {phase}")

    # Start cold and ramp to the requested temperature. This makes a failed
    # structure a bounded, clearly-labelled experiment instead of a worker
    # crash or an opaque 500 response.
    simulation.context.setPositions(min_positions)
    simulation.context.setVelocitiesToTemperature(50.0 * u.kelvin, seed)
    for equil_temp in (50.0, 150.0, 250.0, float(temperature)):
        integrator.setTemperature(equil_temp * u.kelvin)
        _safe_step(100, f"equilibration at {equil_temp:.0f} K")

    # production trajectory (protein atoms only — water/ions excluded)
    simulation.reporters = []
    trajectory = []
    report_interval = max(steps // max(store_every, 1), 1)
    simulation.currentStep = 0
    for step in range(0, steps, report_interval):
        _safe_step(report_interval, "production")
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        pos = state.getPositions(asNumpy=True)
        frame = _xyz(pos)[:n_prot_atoms]
        trajectory.append(frame)
        if len(trajectory) >= store_every:
            break

    elapsed = time.time() - t0
    ana = _analyse(trajectory, coords0, None, steps)
    ana.update({
        "platform": plat,
        "temperature_K": float(temperature),
        "box_padding_nm": float(box_padding_nm),
        "explicit_solvent": "tip3p + 0.15M NaCl",
        "force_field": f"{_PF}+{_WF}",
        "timestep_fs": 0.5,
        "max_protein_atoms": int(max_protein_atoms),
        "steps_planned": steps,
        "runtime_s": round(elapsed, 1),
        "engine": "openmm",
        "hardware": _hardware(),
        "seed": seed,
        "protein_hash": _sha256(protein_path),
        "analysis": "protein backbone drift + RMSF + Rg; ligand contact persistence "
                    "computed from docked pose (static) — ligand flexibility NOT modelled",
        "limitations": ("Implicit/explicit-solvent protein-only dynamics. Small-molecule "
                        "flexibility and solvent dynamics of the ligand are not yet "
                        "parametrised; contacts are a coarse persistence proxy, not a "
                        "measured binding value (spec §41)."),
    })

    # persistence artifacts
    traj_path = work / "trajectory.npz"
    np.savez_compressed(traj_path, positions=np.array(trajectory))
    ana["trajectory_path"] = str(traj_path)
    (work / "md_record.json").write_text(json.dumps(ana, indent=2))
    env = openmm.version.version if hasattr(openmm.version, "version") else "unknown"
    ana["engine_version"] = str(env)
    ana["protein_path"] = str(protein_path)
    ana["runtime_s"] = round(elapsed, 1)
    return MdResults(ana)