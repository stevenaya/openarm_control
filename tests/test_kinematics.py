"""Regression tests for OpenArm differential IK timing and nullspace control."""

from __future__ import annotations

import argparse
import pathlib
import tempfile
import unittest
from unittest import mock

import mink
import mujoco
import numpy as np
import openarm_mujoco.v2 as openarm_mujoco

from openarm_control import (
    ArmSetup,
    IKParams,
    Kinematics,
    RecoverableConfigurationLimit,
    ik_params_from_args,
    read_ee_pose,
    register_ik_args,
)
from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from openarm_control.kinematics import (
    _arm_qpos_indices,
    _dof_indices_for_qpos,
)
from openarm_control.nullspace_posture_task import (
    NullspacePostureTask,
    smoothstep_activation,
    structural_nullspace_direction,
)


def _setup(mode: str = "bimanual") -> ArmSetup:
    return ArmSetup.from_args(
        xml=openarm_mujoco.openarm_cell_xml(),
        mode=mode,
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe="home",
    )


class NullspaceMathTest(unittest.TestCase):
    """Exercise the pure SVD and singularity-activation helpers."""

    def test_structural_direction_is_null_and_sign_continuous(self) -> None:
        rng = np.random.default_rng(7)
        jacobian = rng.normal(size=(6, 7))

        direction, singular_values = structural_nullspace_direction(jacobian)
        aligned, _ = structural_nullspace_direction(jacobian, previous=-direction)

        self.assertEqual(singular_values.shape, (6,))
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)
        self.assertLess(float(np.linalg.norm(jacobian @ direction)), 1e-12)
        self.assertGreater(float(aligned @ (-direction)), 1.0 - 1e-12)

    def test_smoothstep_activation_has_clamped_flat_endpoints(self) -> None:
        self.assertEqual(smoothstep_activation(0.0, 0.02, 0.05), 0.0)
        self.assertEqual(smoothstep_activation(0.02, 0.02, 0.05), 0.0)
        self.assertAlmostEqual(smoothstep_activation(0.035, 0.02, 0.05), 0.5)
        self.assertEqual(smoothstep_activation(0.05, 0.02, 0.05), 1.0)
        self.assertEqual(smoothstep_activation(1.0, 0.02, 0.05), 1.0)


class NullspaceTaskTest(unittest.TestCase):
    """Exercise the task against the actual OpenArm MuJoCo model."""

    def test_task_is_nullspace_only_and_caps_home_return_speed(self) -> None:
        setup = _setup("right")
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        frame_task = mink.FrameTask(
            frame_name="right_ee_control_point",
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
        )
        frame_task.set_target_from_configuration(configuration)
        dof_indices = _dof_indices_for_qpos(
            setup.model, _arm_qpos_indices(setup, "right")
        )
        home_qpos = configuration.q
        task = NullspacePostureTask(
            model=setup.model,
            frame_task=frame_task,
            dof_indices=dof_indices,
            home_qpos=home_qpos,
            cost=0.3,
            dt=0.0004,
            return_rate=100.0,
            max_speed=0.5,
            singularity_low=0.0,
            singularity_high=1e-9,
            characteristic_length=0.3,
        )

        task.compute_qp_objective(configuration)
        initial_state = task.last_state
        self.assertIsNotNone(initial_state)
        assert initial_state is not None

        displaced_qpos = configuration.q
        tangent = np.zeros(setup.model.nv)
        tangent[dof_indices] = initial_state.direction
        mujoco.mj_integratePos(setup.model, displaced_qpos, tangent, 0.5)
        configuration.update(q=displaced_qpos)

        objective = task.compute_qp_objective(configuration)
        state = task.last_state
        self.assertIsNotNone(state)
        assert state is not None

        self.assertLess(state.jacobian_residual, 1e-10)
        self.assertLessEqual(abs(state.return_speed), 0.5)
        self.assertAlmostEqual(abs(state.displacement), 0.5 * 0.0004)
        self.assertAlmostEqual(state.effective_cost, np.sqrt(state.activation) * 0.3)

        z_full = np.zeros(setup.model.nv)
        z_full[dof_indices] = state.direction
        expected_h = 0.3**2 * state.activation * np.outer(z_full, z_full)
        expected_c = -(0.3**2) * state.activation * state.displacement * z_full
        np.testing.assert_allclose(objective.H, expected_h, atol=1e-12)
        np.testing.assert_allclose(objective.c, expected_c, atol=1e-12)

    def test_sync_does_not_change_fixed_home_targets(self) -> None:
        setup = _setup()
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=10,
                posture_cost=0.01,
                nullspace_cost=0.3,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        posture_home = solver._posture_task.target_q.copy()
        nullspace_homes = {
            side: task._home_qpos.copy()
            for side, task in solver._nullspace_tasks.items()
        }

        measured = np.linspace(-0.2, 0.2, 16, dtype=np.float32)
        kinematics.sync(measured)

        np.testing.assert_array_equal(solver._posture_task.target_q, posture_home)
        for side, task in solver._nullspace_tasks.items():
            np.testing.assert_array_equal(task._home_qpos, nullspace_homes[side])


class SolverTimingTest(unittest.TestCase):
    """Verify that ten substeps represent exactly one outer control period."""

    def test_cli_uses_tick_period_and_unscaled_velocity_limits(self) -> None:
        parser = argparse.ArgumentParser()
        register_ik_args(parser)
        args = parser.parse_args(
            [
                "--tick-hz",
                "250",
                "--limit-velocity",
                "--joint-limit-recovery-velocity-scale",
                "1.2",
            ]
        )

        params = ik_params_from_args(args)

        self.assertAlmostEqual(params.dt, 0.004)
        self.assertEqual(params.joint_limit_recovery_velocity_scale, 1.2)
        assert params.velocity_limits is not None
        for side in ("left", "right"):
            for index, expected in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S):
                self.assertEqual(
                    params.velocity_limits[f"openarm_{side}_joint{index + 1}"],
                    expected,
                )

    def test_custom_arm_velocity_limits_override_defaults(self) -> None:
        custom_caps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / "ik.yaml"
            config_path.write_text(
                "arm_velocity_limits:\n"
                + "".join(f"  - {cap}\n" for cap in custom_caps),
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            register_ik_args(parser)
            args = parser.parse_args(
                ["--limit-velocity", "--config", str(config_path)]
            )

            params = ik_params_from_args(args)

        assert params.velocity_limits is not None
        for side in ("left", "right"):
            for index, expected in enumerate(custom_caps):
                self.assertEqual(
                    params.velocity_limits[f"openarm_{side}_joint{index + 1}"],
                    expected,
                )

    def test_driver_delta_limits_are_not_accepted_as_ik_velocity_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / "driver.yaml"
            config_path.write_text(
                "joint_delta_position_limits: [1, 1, 1, 1, 1, 1, 1, 1]\n",
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            register_ik_args(parser)
            args = parser.parse_args(
                ["--limit-velocity", "--config", str(config_path)]
            )

            with self.assertRaisesRegex(ValueError, "arm_velocity_limits"):
                ik_params_from_args(args)

    def test_substep_dt_and_velocity_limits_stay_in_physical_units(self) -> None:
        setup = _setup()
        per_joint_caps = np.linspace(0.2, 0.8, 7)
        velocity_limits = {
            f"openarm_{side}_joint{index + 1}": float(cap)
            for side in ("left", "right")
            for index, cap in enumerate(per_joint_caps)
        }
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=10,
                posture_cost=0.0,
                nullspace_cost=0.0,
                velocity_limits=velocity_limits,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        self.assertAlmostEqual(solver._substep_dt, 0.0004)

        velocity_limit = solver._limits[-1]
        self.assertIsInstance(velocity_limit, RecoverableConfigurationLimit)
        self.assertEqual(len(solver._limits), 1)
        np.testing.assert_allclose(
            np.sort(velocity_limit.limit),
            np.sort(np.tile(per_joint_caps, 2)),
        )

        velocity = np.zeros(setup.model.nv)
        for side in ("right", "left"):
            dof_indices = _dof_indices_for_qpos(
                setup.model, _arm_qpos_indices(setup, side)
            )
            velocity[dof_indices] = per_joint_caps
        qpos_before = solver._config.q

        kinematics.set_target("right", setup.read_ee_pose("right"))
        kinematics.set_target("left", setup.read_ee_pose("left"))
        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            return_value=velocity,
        ) as solve_ik:
            result = kinematics.solve()

        self.assertIsNotNone(result)
        self.assertEqual(solve_ik.call_count, 10)
        for call in solve_ik.call_args_list:
            self.assertAlmostEqual(call.args[2], 0.0004)

        expected_delta = velocity * 0.004
        np.testing.assert_allclose(
            solver._config.q - qpos_before,
            expected_delta,
            atol=1e-12,
        )


class RecoverableConfigurationLimitTest(unittest.TestCase):
    """Exercise the joint position/velocity intersection around overshoot."""

    def _solver_and_limit(
        self,
    ) -> tuple[Kinematics, RecoverableConfigurationLimit]:
        setup = _setup()
        velocity_limits = {
            f"openarm_{side}_joint{index + 1}": float(cap)
            for side in ("left", "right")
            for index, cap in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
        }
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=10,
                posture_cost=0.0,
                nullspace_cost=0.0,
                velocity_limits=velocity_limits,
                joint_limit_recovery_velocity_scale=1.1,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        self.assertEqual(len(solver._limits), 1)
        limit = solver._limits[0]
        self.assertIsInstance(limit, RecoverableConfigurationLimit)
        assert isinstance(limit, RecoverableConfigurationLimit)
        return kinematics, limit

    def test_inside_range_uses_normal_velocity_cap(self) -> None:
        kinematics, limit = self._solver_and_limit()
        solver = kinematics._ik
        assert solver is not None
        row = 0
        q = solver._config.q
        q[limit.qpos_indices[row]] = 0.5 * (
            limit.lower[row] + limit.upper[row]
        )
        solver._config.update(q=q)

        inequalities = limit.compute_qp_inequalities(
            solver._config, solver._substep_dt
        )
        assert inequalities.h is not None
        count = limit.indices.size
        step_lower = -inequalities.h[count:]
        step_upper = inequalities.h[:count]
        normal_step = limit.limit[row] * solver._substep_dt

        self.assertAlmostEqual(step_lower[row], -normal_step)
        self.assertAlmostEqual(step_upper[row], normal_step)

    def test_outside_range_recovers_with_wider_feasible_cap(self) -> None:
        kinematics, limit = self._solver_and_limit()
        solver = kinematics._ik
        assert solver is not None
        row = 0
        qpos_index = limit.qpos_indices[row]
        recovery_step = (
            limit.limit[row]
            * solver._substep_dt
            * limit.recovery_velocity_scale
        )

        for q_value, expected_step in (
            (limit.lower[row] - 0.01, recovery_step),
            (limit.upper[row] + 0.01, -recovery_step),
        ):
            q = solver._config.q
            q[qpos_index] = q_value
            solver._config.update(q=q)
            inequalities = limit.compute_qp_inequalities(
                solver._config, solver._substep_dt
            )
            assert inequalities.h is not None
            count = limit.indices.size
            step_lower = -inequalities.h[count:]
            step_upper = inequalities.h[:count]

            self.assertLessEqual(step_lower[row], step_upper[row])
            self.assertAlmostEqual(step_lower[row], expected_step)
            self.assertAlmostEqual(step_upper[row], expected_step)

    def test_real_qp_recovers_from_joint_limit_overshoot(self) -> None:
        kinematics, limit = self._solver_and_limit()
        solver = kinematics._ik
        assert solver is not None
        row = 0
        qpos_index = limit.qpos_indices[row]
        q = solver._config.q
        q[qpos_index] = limit.lower[row] - 0.001
        solver._config.update(q=q)
        initial_q = float(solver._config.q[qpos_index])

        for side in kinematics.setup.sides:
            pose = read_ee_pose(
                solver._config.data,
                kinematics.setup.frame_ids[side],
                kinematics.setup.frame_types[side],
            )
            kinematics.set_target(side, pose)

        result = kinematics.solve()

        self.assertIsNotNone(result)
        self.assertGreater(float(solver._config.q[qpos_index]), initial_q)


class SolverRecoveryTest(unittest.TestCase):
    """Verify that failed solves do not corrupt bimanual solver state."""

    def test_failure_rolls_back_and_waits_for_fresh_target_pair(self) -> None:
        setup = _setup()
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=3,
                posture_cost=0.0,
                nullspace_cost=0.0,
            ),
        )
        solver = kinematics._ik
        assert solver is not None

        target_right = setup.read_ee_pose("right")
        target_left = setup.read_ee_pose("left")
        kinematics.set_target("right", target_right)
        kinematics.set_target("left", target_left)
        self.assertTrue(kinematics.ready())

        velocity = np.zeros(setup.model.nv)
        right_dofs = _dof_indices_for_qpos(
            setup.model, _arm_qpos_indices(setup, "right")
        )
        velocity[right_dofs[0]] = 0.5
        qpos_before = solver._config.q.copy()

        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            side_effect=[velocity, mink.exceptions.NoSolutionFound("daqp")],
        ) as solve_ik:
            result = kinematics.solve()

        self.assertIsNone(result)
        self.assertEqual(solve_ik.call_count, 2)
        np.testing.assert_array_equal(solver._config.q, qpos_before)
        self.assertFalse(kinematics.ready())

        kinematics.set_target("right", target_right)
        self.assertFalse(kinematics.ready())
        kinematics.set_target("left", target_left)
        self.assertTrue(kinematics.ready())

        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            return_value=np.zeros(setup.model.nv),
        ):
            recovered = kinematics.solve()

        self.assertIsNotNone(recovered)
        self.assertFalse(kinematics.ready())


if __name__ == "__main__":
    unittest.main()
