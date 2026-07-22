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
    LowerBoundBrakingLimit,
    RecoverableConfigurationLimit,
    ik_params_from_args,
    read_ee_pose,
    register_ik_args,
)
from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from openarm_control.kinematics import (
    _arm_qpos_indices,
    _dof_indices_for_qpos,
    _elbow_joint_indices,
)
from openarm_control.nullspace_posture_task import (
    NullspacePostureTask,
    smoothstep_activation,
    structural_nullspace_direction,
)
from openarm_control.soft_limit_task import SoftLimitTask


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


class SoftLimitTaskTest(unittest.TestCase):
    """Exercise the one-sided joint4 guard independently of the IK solver."""

    def _task_and_configuration(
        self,
    ) -> tuple[SoftLimitTask, mink.Configuration, int]:
        setup = _setup("right")
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        elbow_qpos, elbow_dof = _elbow_joint_indices(setup, "right")
        task = SoftLimitTask(
            model=setup.model,
            joint_qpos_index=elbow_qpos,
            joint_dof_index=elbow_dof,
            cost=2.0,
            dt=0.01,
            limit=0.1,
            max_speed=0.05,
        )
        qpos = configuration.q
        qpos[elbow_qpos] = 0.0
        configuration.update(q=qpos)
        return task, configuration, elbow_dof

    def test_straight_arm_requests_only_slow_joint4_bending(self) -> None:
        task, configuration, elbow_dof = self._task_and_configuration()

        objective = task.compute_qp_objective(configuration)

        selector = np.zeros(configuration.model.nv)
        selector[elbow_dof] = 1.0
        expected_h = 2.0**2 * np.outer(selector, selector)
        expected_c = -(2.0**2) * 0.0005 * selector
        np.testing.assert_allclose(objective.H, expected_h, atol=1e-12)
        np.testing.assert_allclose(objective.c, expected_c, atol=1e-12)

    def test_task_turns_off_at_limit(self) -> None:
        task, configuration, _ = self._task_and_configuration()
        qpos = configuration.q
        qpos[task._joint_qpos_index] = task._limit
        configuration.update(q=qpos)
        objective = task.compute_qp_objective(configuration)
        np.testing.assert_array_equal(objective.H, np.zeros_like(objective.H))
        np.testing.assert_array_equal(objective.c, np.zeros_like(objective.c))

    def test_solver_builds_one_guard_for_each_active_arm(self) -> None:
        setup = _setup()
        kinematics = Kinematics(
            setup,
            IKParams(
                posture_cost=0.0,
                nullspace_cost=0.0,
                elbow_soft_limit_cost=2.0,
            ),
        )
        solver = kinematics._ik
        assert solver is not None

        self.assertEqual(set(solver._elbow_soft_limit_tasks), {"left", "right"})
        for side, task in solver._elbow_soft_limit_tasks.items():
            expected_qpos, expected_dof = _elbow_joint_indices(setup, side)
            self.assertEqual(task._joint_qpos_index, expected_qpos)
            self.assertEqual(task._joint_dof_index, expected_dof)


class LowerBoundBrakingLimitTest(unittest.TestCase):
    """Exercise the preventive joint4 stopping-distance constraint."""

    def _limit_and_configuration(
        self,
    ) -> tuple[LowerBoundBrakingLimit, mink.Configuration]:
        setup = _setup("right")
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        elbow_qpos, elbow_dof = _elbow_joint_indices(setup, "right")
        limit = LowerBoundBrakingLimit(
            model=setup.model,
            joint_qpos_index=elbow_qpos,
            joint_dof_index=elbow_dof,
            guard_position=0.08,
            max_deceleration=20.0,
            max_velocity=3.14,
        )
        return limit, configuration

    def _allowed_displacement(self, position: float) -> float:
        limit, configuration = self._limit_and_configuration()
        q = configuration.q
        q[limit.joint_qpos_index] = position
        configuration.update(q=q)

        inequalities = limit.compute_qp_inequalities(configuration, dt=0.0004)

        assert inequalities.G is not None
        assert inequalities.h is not None
        self.assertEqual(inequalities.G[0, limit.joint_dof_index], -1.0)
        return float(inequalities.h[0])

    def test_far_from_guard_uses_joint_velocity_cap(self) -> None:
        self.assertAlmostEqual(
            self._allowed_displacement(0.7),
            3.14 * 0.0004,
        )

    def test_near_guard_uses_stopping_distance_velocity(self) -> None:
        margin = 0.05
        expected_velocity = np.sqrt(2.0 * 20.0 * margin)
        self.assertAlmostEqual(
            self._allowed_displacement(0.08 + margin),
            expected_velocity * 0.0004,
        )

    def test_at_or_below_guard_forbids_further_approach(self) -> None:
        self.assertEqual(self._allowed_displacement(0.08), 0.0)
        self.assertEqual(self._allowed_displacement(0.04), 0.0)

    def test_guard_must_be_inside_physical_joint_range(self) -> None:
        setup = _setup("right")
        elbow_qpos, elbow_dof = _elbow_joint_indices(setup, "right")
        with self.assertRaisesRegex(ValueError, "physical joint range"):
            LowerBoundBrakingLimit(
                model=setup.model,
                joint_qpos_index=elbow_qpos,
                joint_dof_index=elbow_dof,
                guard_position=-0.01,
                max_deceleration=20.0,
                max_velocity=3.14,
            )

    def test_solver_builds_one_limit_for_each_active_arm(self) -> None:
        setup = _setup()
        kinematics = Kinematics(
            setup,
            IKParams(
                posture_cost=0.0,
                nullspace_cost=0.0,
                elbow_braking_guard_angle=0.08,
                elbow_braking_acceleration=20.0,
            ),
        )
        solver = kinematics._ik
        assert solver is not None

        self.assertEqual(set(solver._elbow_braking_limits), {"left", "right"})
        self.assertEqual(len(solver._limits), 3)
        for side, limit in solver._elbow_braking_limits.items():
            expected_qpos, expected_dof = _elbow_joint_indices(setup, side)
            self.assertEqual(limit.joint_qpos_index, expected_qpos)
            self.assertEqual(limit.joint_dof_index, expected_dof)

    def test_outward_reach_brakes_before_elbow_guard(self) -> None:
        setup = _setup("right")
        velocity_limits = {
            f"openarm_{side}_joint{index + 1}": float(cap)
            for side in ("left", "right")
            for index, cap in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
        }
        kinematics = Kinematics(
            setup,
            IKParams(
                position_cost=10.0,
                orientation_cost=1.0,
                lm_damping=0.02,
                damping=0.1,
                posture_cost=0.0,
                dt=0.004,
                max_iters=10,
                velocity_limits=velocity_limits,
                joint_limit_recovery_velocity_scale=1.0,
                nullspace_cost=10.0,
                nullspace_return_rate=0.8,
                nullspace_max_speed=0.8,
                nullspace_singularity_low=0.02,
                nullspace_singularity_high=0.05,
                nullspace_characteristic_length=0.3,
                elbow_braking_guard_angle=0.08,
                elbow_braking_acceleration=20.0,
            ),
        )
        initial_right = np.array([1.224145, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0])
        left_home, _ = setup.joint_resolver.get_driver(setup.data.qpos, "left")
        initial_pose = kinematics.fk("right", initial_right)
        kinematics.sync(
            np.concatenate([initial_right, left_home, np.zeros(1)]).astype(np.float32)
        )

        commands: list[np.ndarray] = []
        for step in range(230):
            target = initial_pose.copy()
            target[0] += 0.05 * 0.004 * (step + 1)
            kinematics.set_target("right", target)
            result = kinematics.solve()
            self.assertIsNotNone(result)
            assert result is not None
            commands.append(result[:7].astype(np.float64))

        command_array = np.asarray(commands)
        elbow_position = command_array[:, 3]
        elbow_velocity = (
            np.diff(np.concatenate([[initial_right[3]], elbow_position])) / 0.004
        )
        elbow_acceleration = np.diff(np.concatenate([[0.0], elbow_velocity])) / 0.004
        braking_region = elbow_position < 0.2

        self.assertGreaterEqual(float(np.min(elbow_position)), 0.08 - 1e-4)
        self.assertLess(
            float(np.max(np.abs(elbow_acceleration[braking_region]))),
            25.0,
        )


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
                "--elbow-soft-limit-cost",
                "2.0",
                "--elbow-soft-limit-angle",
                "0.08",
                "--elbow-soft-limit-max-speed",
                "0.2",
                "--elbow-braking-guard-angle",
                "0.08",
                "--elbow-braking-acceleration",
                "20.0",
            ]
        )

        params = ik_params_from_args(args)

        self.assertAlmostEqual(params.dt, 0.004)
        self.assertEqual(params.joint_limit_recovery_velocity_scale, 1.2)
        self.assertEqual(params.elbow_soft_limit_cost, 2.0)
        self.assertEqual(params.elbow_soft_limit_angle, 0.08)
        self.assertEqual(params.elbow_soft_limit_max_speed, 0.2)
        self.assertEqual(params.elbow_braking_guard_angle, 0.08)
        self.assertEqual(params.elbow_braking_acceleration, 20.0)
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
            args = parser.parse_args(["--limit-velocity", "--config", str(config_path)])

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
            args = parser.parse_args(["--limit-velocity", "--config", str(config_path)])

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
        q[limit.qpos_indices[row]] = 0.5 * (limit.lower[row] + limit.upper[row])
        solver._config.update(q=q)

        inequalities = limit.compute_qp_inequalities(solver._config, solver._substep_dt)
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
            limit.limit[row] * solver._substep_dt * limit.recovery_velocity_scale
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
