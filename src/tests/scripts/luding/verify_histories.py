#!/usr/bin/env python3
"""Integration checks for the single-history Luding model."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile


def check_close(actual, expected, tolerance, label):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")


def input_text(fixed=False, phi=0.1, f0=0.0, restitution=1.0,
               drop=False, elastic=False):
    k1 = 2208.490298592278 if drop else 1000.0
    k2 = k1 if elastic else 5 * k1
    kc = 0.0 if elastic else 0.2 * k1
    top = 0.020 if drop else 0.015
    text = f"""units si
atom_style granular
atom_modify map array
boundary f f f
newton off
communicate single vel yes
region box block -0.02 0.02 -0.02 0.02 -0.01 0.06 units box
create_box 1 box
create_atoms 1 single 0 0 0.005 units box
create_atoms 1 single 0 0 {top} units box
set group all diameter 0.01
set group all density 2500
group fixed id 1
group moving id 2
neighbor 0.001 bin
neigh_modify delay 0 every 1 check yes
fix mu all property/global coefficientFriction peratomtypepair 1 0
fix eps all property/global coefficientRestitution peratomtypepair 1 {restitution}
fix pk1 all property/global k1 peratomtypepair 1 {k1}
fix pk2 all property/global k2Max peratomtypepair 1 {k2}
fix pkc all property/global kc peratomtypepair 1 {kc}
fix pf0 all property/global F0 peratomtypepair 1 {f0}
"""
    # Deliberately omit phiF in fixed-k2 tests.
    if not fixed:
        text += f"fix pphi all property/global phiF peratomtypepair 1 {phi}\n"
    text += f"""pair_style gran model luding tangential history fixedK2 {'on' if fixed else 'off'}
pair_coeff * *
fix hold fixed freeze
timestep 1e-6
variable s equal step
variable d equal 0.015-z[2]
variable vz equal vz[2]
variable f equal fz[2]
variable zfixed equal z[1]
fix data all print 1 "${{s}},${{d}},${{vz}},${{f}},${{zfixed}}" file history.csv screen no title "step,delta,vz,force,zfixed"
thermo 10000
thermo_style custom step atoms
thermo_modify lost error norm no
"""
    if drop:
        text += "fix gravity moving gravity 9.81 vector 0 0 -1\n"
        text += "fix integr moving nve/sphere\nrun 40000\n"
    else:
        peak_steps = 1000 if phi == 0.1 and not fixed else 200
        # Unsaturated tests use a larger hardening scale below.
        segments = [(peak_steps, -1), (peak_steps - 20, 1),
                    (80, -1), (220, 1), (320, -1)]
        for count, speed in segments:
            text += f"fix mover moving move linear 0 0 {speed} units box\n"
            text += f"run {count}\nunfix mover\n"
    return text


def run_case(binary, output, name, **parameters):
    folder = output / name
    folder.mkdir()
    (folder / "in.test").write_text(input_text(**parameters))
    with (folder / "console.log").open("w") as stream:
        process = subprocess.run(
            [str(binary), "-in", "in.test", "-log", "log.liggghts"],
            cwd=folder, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise AssertionError(f"{name} failed; see {folder / 'console.log'}")
    with (folder / "history.csv").open() as stream:
        rows = [{k: float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]
    assert rows and all(math.isfinite(v) for row in rows for v in row.values())
    assert all(row["zfixed"] == 0.005 for row in rows)
    log = (folder / "log.liggghts").read_text()
    assert "ERROR" not in log and "Dangerous builds = 0" in log
    return rows


def prescribed_checks(rows, fixed, phi):
    k1, k2max, kc = 1000.0, 5000.0, 200.0
    limit = 5 / 4 * phi * 0.005
    peak_steps = 200 if fixed or phi != 0.1 else 1000
    loading = [r for r in rows if r["step"] <= peak_steps]
    peak = loading[-1]["delta"]
    k2 = k2max if fixed or peak >= limit else k1 + (k2max - k1) * peak / limit
    anchor = peak if fixed or peak < limit else limit
    max_error = 0.0
    for row in loading:
        d = row["delta"]
        expected = k1 * d if fixed or d < limit else k2max * d - (k2max - k1) * limit
        max_error = max(max_error, abs(row["force"] - expected))
    turn_step = 2 * peak_steps - 20
    unloading = [r for r in rows if peak_steps < r["step"] <= turn_step]
    for row in unloading:
        d = row["delta"]
        expected = max(k2 * d - (k2 - k1) * anchor, -kc * d)
        max_error = max(max_error, abs(row["force"] - expected))
    check_close(max_error, 0, 1e-9, "prescribed loading/unloading force")

    # Adhesion reversal must start a k2 line through the turning point.
    turn = unloading[-1]["delta"]
    if fixed:
        reload_k2 = k2max
    else:
        # Analytic solution of k2(h)*(turn-h)+k1*h = -kc*turn.
        h = (turn + math.sqrt(turn**2 + 4 * (k1 + kc) * turn * limit /
                             (k2max - k1))) / 2
        assert h < limit  # Includes leaving a previously saturated state.
        reload_k2 = k1 + (k2max - k1) * h / limit
    reload_rows = [
        r for r in rows if turn_step < r["step"] <= turn_step + 10]
    for row in reload_rows:
        expected = min(k1 * row["delta"],
                       -kc * turn + reload_k2 * (row["delta"] - turn))
        check_close(row["force"], expected, 1e-9, "adhesion-to-reload line")

    # The last segment crosses separation and creates a fresh contact.
    last_start = turn_step + 80 + 220
    gap = [r for r in rows if r["delta"] < -2e-6]
    assert gap
    assert all(abs(r["force"]) < 1e-12 for r in gap)
    recontact = [r for r in rows if r["step"] > last_start and r["delta"] > 2e-6]
    assert recontact
    for row in recontact:
        check_close(row["force"], k1 * row["delta"], 1e-9, "fresh-contact loading")
    return {"max_force_error_N": max_error, "reload_stiffness_N_m": reload_k2}


def interpolate(a, b):
    weight = -a["delta"] / (b["delta"] - a["delta"])
    return {k: a[k] + weight * (b[k] - a[k]) for k in a}


def drop_checks(rows, hard_core, elastic=False):
    mass = 2500 * 4 / 3 * math.pi * 0.005**3
    gravity = 9.81
    vin = math.sqrt(2 * gravity * 0.005)
    kinetic = 0.5 * mass * vin**2
    k1 = 2208.490298592278
    k2 = k1 if elastic else 5 * k1
    kc = 0 if elastic else 0.2 * k1
    limit = 5 / 4 * 0.01 * 0.005
    if hard_core:
        intercept = (k2 - k1) * limit + mass * gravity
        peak = (intercept + math.sqrt(intercept**2 -
                k2 * ((k2 - k1) * limit**2 - mass * vin**2))) / k2
        loss = 0.5 * (k2 - k1) * (k1 + kc) / (k2 + kc) * limit**2
    else:
        peak = (mass * gravity + math.sqrt((mass * gravity)**2 +
                k1 * mass * vin**2)) / k1
        loss = 0.5 * (k2 - k1) * (k1 + kc) / (k2 + kc) * peak**2
    expected_e = math.sqrt(1 - loss / kinetic)
    start = next(i for i, row in enumerate(rows) if row["delta"] > 0)
    end = next(i for i in range(start + 1, len(rows)) if rows[i]["delta"] <= 0)
    incoming = interpolate(rows[start - 1], rows[start])
    outgoing = interpolate(rows[end - 1], rows[end])
    measured_peak = max(r["delta"] for r in rows[start:end])
    measured_e = outgoing["vz"] / -incoming["vz"]
    check_close(measured_peak, peak, peak * 1e-5, "analytical peak overlap")
    check_close(measured_e, expected_e, 1e-5, "analytical restitution")
    return {"theoretical_peak_m": peak, "measured_peak_m": measured_peak,
            "theoretical_restitution": expected_e, "measured_restitution": measured_e}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="luding-histories-"))
    output.mkdir(exist_ok=True, parents=True)
    print(f"Test output: {output}", flush=True)
    report = {}
    paths = {}
    for name, fixed, phi in [("fixed", True, 0.1), ("unsaturated", False, 0.2),
                             ("saturated", False, 0.1)]:
        rows = run_case(binary, output, name, fixed=fixed, phi=phi)
        report[name] = prescribed_checks(rows, fixed, phi)
        paths[name] = rows

    offset = run_case(binary, output, "force_offset", fixed=True, f0=0.02)
    for original, row in zip(paths["fixed"], offset):
        expected = original["force"] - 0.02 if row["delta"] > 1e-9 else original["force"]
        if abs(row["delta"]) > 1e-9:
            check_close(row["force"], expected, 1e-9, "positive F0 attraction")
    report["force_offset"] = "positive F0 subtracts from contact force on every branch"

    mass = 2500 * 4 / 3 * math.pi * 0.005**3
    factor = 4 * math.log(0.8)**2 / (math.log(0.8)**2 + math.pi**2)
    damped = run_case(binary, output, "damping", fixed=False,
                      phi=0.1, restitution=0.8)
    for row in damped:
        if row["step"] > 1980:
            break
        if row["step"] > 1000:
            hard = 5000 * row["delta"] - 4000 * 0.000625
            adhesion = -200 * row["delta"]
            stiffness = 5000 if hard >= adhesion else 200
            expected = max(hard, adhesion) - math.sqrt(mass * stiffness * factor)
            check_close(row["force"], expected, 1e-8, "saturated unloading damping")
            continue
        # At the exact saturation point the dashpot changes discontinuously;
        # independently computed overlaps can round to opposite sides.
        if abs(row["delta"] - 0.000625) < 1e-9:
            continue
        stiffness = 1000 if row["delta"] < 0.000625 else 5000
        force = (1000 * row["delta"] if stiffness == 1000 else
                 5000 * row["delta"] - 4000 * 0.000625)
        expected = force + math.sqrt(mass * stiffness * factor)
        check_close(row["force"], expected, 1e-8, "active-branch loading damping")
    report["damping"] = "k1 before saturation; k2Max hard-core and kc adhesion damping"

    for name, fixed, elastic in [
            ("drop_single_variable", False, False),
            ("drop_single_fixed", True, False),
            ("drop_elastic", True, True)]:
        rows = run_case(binary, output, name, fixed=fixed,
                        phi=0.01, drop=True, elastic=elastic)
        report[name] = drop_checks(rows, not fixed, elastic)

    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("PASS: all Luding history integration checks")


if __name__ == "__main__":
    main()
