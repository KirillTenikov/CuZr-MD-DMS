from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1] / "scripts" / "run"
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import mddms_branch_runner as generic
import paper2_revision_runner as legacy


HISTORICAL_STAGE03 = """units metal
atom_style atomic
boundary p p p
newton on
read_data       02_after_equilibrate_nvt.data

pair_style mliap unified /tmp/MACE_C.model-mliap_lammps.pt 0
pair_coeff * * Cu Zr
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

timestep 0.00100000
change_box all triclinic
reset_timestep 0
velocity        all create 300.000000 446 mom yes rot yes dist gaussian
thermo 500
thermo_style custom step time temp pe ke etotal press pxx pyy pzz pxy lx ly lz xy
variable gamma0 equal 0.010000000000
variable period equal 20.000000000000
variable omega equal 2.0*3.1415926535897931/v_period
variable gamma equal v_gamma0*sin(v_omega*time)
variable gammadot equal v_gamma0*v_omega*cos(v_omega*time)
variable xy_target equal v_gamma*ly
variable xy_rate equal v_gammadot*ly
fix thermostat all nvt temp 300.000000 300.000000 0.100000
fix deform all deform 1 xy variable v_xy_target v_xy_rate remap x
run 120000
unfix deform
unfix thermostat
"""


def make_protocol(**overrides) -> legacy.Protocol:
    values = {
        "preset": "pressure_relaxed",
        "model_alias": "mace_c",
        "natoms": 4000,
        "cu_fraction": 0.50,
        "density_g_cm3": 7.20,
        "seed": 42,
        "timestep_ps": 0.001,
        "temperature_high_K": 3000.0,
        "temperature_low_K": 300.0,
        "pressure_bar": 0.0,
        "strain_amplitude": 0.005,
        "tdamp_ps": 1.0,
        "pdamp_ps": 1.0,
        "mddms_period_ps": 20.0,
        "mddms_cycles": 6,
        "thermo_every_steps": None,
        "stress_every_steps": None,
        "dump_every_steps": 1000,
        "stress_sign": -1.0,
        "lmp_command": "lmp",
        "checkpoint_every_steps": 10000,
    }
    values.update(overrides)
    return legacy.Protocol(**values)


class GenericBranchRunnerTests(unittest.TestCase):
    def test_historical_start_mode_is_byte_preserving_before_checkpoint_patch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stage = root / legacy.MDDMS_STAGE
            stage.write_text(HISTORICAL_STAGE03, encoding="utf-8")
            generated = generic._apply_start_mode(root, generic.START_DATA_RECREATE)
            self.assertEqual(stage.read_text(encoding="utf-8"), HISTORICAL_STAGE03)
            self.assertEqual(generated.read_text(encoding="utf-8"), HISTORICAL_STAGE03)

    def test_restart_start_mode_preserves_protocol_but_removes_velocity_recreation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stage = root / legacy.MDDMS_STAGE
            stage.write_text(HISTORICAL_STAGE03, encoding="utf-8")
            (root / "02_after_equilibrate_nvt.restart").write_bytes(b"restart-placeholder")
            generic._apply_start_mode(root, generic.START_RESTART_PRESERVE)
            text = stage.read_text(encoding="utf-8")
            self.assertIn("read_restart    02_after_equilibrate_nvt.restart", text)
            self.assertNotIn("read_data       02_after_equilibrate_nvt.data", text)
            self.assertNotIn("velocity        all create", text)
            self.assertIn("variable gamma0 equal 0.010000000000", text)
            self.assertIn("variable period equal 20.000000000000", text)
            self.assertIn("fix thermostat all nvt temp 300.000000 300.000000 0.100000", text)
            self.assertIn("fix deform all deform 1 xy variable v_xy_target v_xy_rate remap x", text)
            self.assertIn("run 120000", text)

    def test_protocol_controls_remain_generator_arguments(self):
        protocol = make_protocol()
        cmd = legacy.generator_command(
            Path("scripts/run/run_mddms_pilot.py"), Path("runs"), "control", protocol
        )
        joined = " ".join(cmd)
        self.assertIn("--cu-fraction 0.5", joined)
        self.assertIn("--strain-amplitude 0.005", joined)
        self.assertIn("--tdamp-ps 1.0", joined)
        self.assertIn("--mddms-period-ps 20.0", joined)
        self.assertIn("--mddms-cycles 6", joined)

    def test_preparation_overrides_are_forwarded_to_historical_generator(self):
        args = Namespace(
            melt_ps=100.0,
            quench_rate_K_per_ps=135.0,
            relax_ps=50.0,
            equilibrate_ps=50.0,
        )
        cmd = generic._generator_command_with_preparation_overrides(
            Path("scripts/run/run_mddms_pilot.py"),
            Path("runs"),
            "p1_prepare",
            make_protocol(),
            args,
        )
        joined = " ".join(cmd)
        self.assertIn("--melt-ps 100.0", joined)
        self.assertIn("--quench-rate-K-per-ps 135.0", joined)
        self.assertIn("--relax-ps 50.0", joined)
        self.assertIn("--equilibrate-ps 50.0", joined)

    def test_stage02_manifest_records_effective_preparation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "02_after_equilibrate_nvt.data").write_text("data\n", encoding="utf-8")
            (root / "02_after_equilibrate_nvt.restart").write_bytes(b"restart")
            metadata = {
                "run_config": {
                    "model_alias": "mace_c",
                    "model_kind": "mace",
                    "model_file": str(root / "missing-model.pt"),
                },
                "preset": {
                    "melt_ps": 100.0,
                    "quench_rate_K_per_ps": 135.0,
                    "relax_ps": 50.0,
                    "equilibrate_ps": 50.0,
                },
                "derived": {
                    "melt_steps": 100000,
                    "quench_ps": 20.0,
                    "quench_steps": 20000,
                    "relax_steps": 50000,
                    "equilibrate_steps": 50000,
                },
            }
            (root / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            manifest_path = generic._write_generic_stage02_manifest(
                root, make_protocol(), "paper3_revision"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prep = manifest["preparation"]
            self.assertEqual(prep["effective"]["melt_ps"], 100.0)
            self.assertEqual(prep["effective"]["quench_rate_K_per_ps"], 135.0)
            self.assertEqual(prep["effective"]["relax_ps"], 50.0)
            self.assertEqual(prep["effective"]["equilibrate_ps"], 50.0)
            self.assertEqual(prep["derived"]["melt_steps"], 100000)
            self.assertEqual(prep["derived"]["quench_steps"], 20000)
            self.assertIsNotNone(prep["source_metadata"]["sha256"])


if __name__ == "__main__":
    unittest.main()
