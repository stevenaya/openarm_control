"""Regression tests for OpenArm differential IK."""

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
    ik_params_from_args,
    pose_to_se3,
    register_ik_args,
)
from openarm_control.arm_joint_limit import ArmConfigurationLimit, ArmJointLimit
from openarm_control.bounded_frame_task import BoundedFrameTask
from openarm_control.config import (
    ARM_JOINT_VELOCITY_LIMITS_RAD_S,
    WORLD_FRAME,
)
from openarm_control.nullspace_posture_task import (
    NullspacePostureTask,
    smoothstep_activation,
    structural_nullspace_direction,
)
from openarm_control.singularity_approach_limit import SingularityApproachLimit


def _setup(
    mode: str = "bimanual",
    *,
    origin_frame: str = WORLD_FRAME,
    origin_frame_type: str = "site",
) -> ArmSetup:
    return ArmSetup.from_args(
        xml=openarm_mujoco.openarm_cell_xml(),
        mode=mode,
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe="home",
        origin_frame=origin_frame,
        origin_frame_type=origin_frame_type,
    )


def _driver_state(setup: ArmSetup) -> np.ndarray:
    values = []
    for side in ("right", "left"):
        joints, gripper = setup.joint_resolver.get_driver(setup.data.qpos, side)
        values.append(np.append(joints, gripper))
    return np.concatenate(values).astype(np.float32)


def _velocity_mapping(*sides: str) -> dict[str, float]:
    return {
        f"openarm_{side}_joint{index + 1}": value
        for side in sides
        for index, value in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
    }


class ArmSetupTest(unittest.TestCase):
    """Verify driver-state mapping respects MuJoCo qpos and tangent-space indices."""

    def test_driver_state_mapping_updates_only_active_arm(self) -> None:
        setup = _setup("right")
        base_qpos = setup.data.qpos.copy()
        driver_qpos = _driver_state(setup).astype(np.float64)
        driver_qvel = np.arange(16, dtype=np.float64) * 0.1
        driver_qpos[:7] += 0.1
        driver_qpos[8:15] += 0.2

        model_qpos, model_qvel = setup.driver_state_to_mujoco(
            driver_qpos,
            driver_qvel,
            base_qpos=base_qpos,
        )

        np.testing.assert_array_equal(
            model_qpos[setup.joint_resolver.arm_qpos_indices("right")],
            driver_qpos[:7],
        )
        np.testing.assert_array_equal(
            model_qpos[setup.joint_resolver.arm_qpos_indices("left")],
            base_qpos[setup.joint_resolver.arm_qpos_indices("left")],
        )
        np.testing.assert_array_equal(
            model_qvel[setup.joint_resolver.arm_dof_indices("right")],
            driver_qvel[:7],
        )
        np.testing.assert_array_equal(
            model_qvel[setup.joint_resolver.arm_dof_indices("left")],
            np.zeros(7),
        )


class ParameterTest(unittest.TestCase):
    """Verify the flat public configuration contract."""

    def test_cli_resolves_tested_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        register_ik_args(parser)
        params = ik_params_from_args(
            parser.parse_args(["--tick-hz", "250", "--limit-velocity"])
        )

        self.assertEqual(params.position_cost, 10.0)
        self.assertEqual(params.orientation_cost, 1.0)
        self.assertEqual(params.lm_damping, 0.01)
        self.assertEqual(params.damping, 0.1)
        self.assertEqual(params.posture_cost, 0.0)
        self.assertEqual(params.max_iters, 5)
        self.assertEqual(params.dt, 0.004)
        self.assertEqual(params.frame_position_error_limit, 0.003)
        self.assertEqual(params.nullspace_cost, 12.0)
        self.assertEqual(params.nullspace_return_rate, 1.6)
        self.assertTrue(params.joint_braking)
        self.assertEqual(params.joint_braking_distance, 0.5)
        self.assertEqual(params.singularity_max_approach_rate, 0.25)
        self.assertEqual(params.kinetic_energy_cost, 3e-5)
        self.assertFalse(hasattr(params, "diag_reg"))
        assert params.velocity_limits is not None
        for side in ("left", "right"):
            for index, expected in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S):
                self.assertEqual(
                    params.velocity_limits[f"openarm_{side}_joint{index + 1}"],
                    expected,
                )

    def test_unexposed_parameters_remain_regular_ik_fields(self) -> None:
        parser = argparse.ArgumentParser()
        register_ik_args(parser)
        options = {
            option for action in parser._actions for option in action.option_strings
        }
        params = IKParams(
            frame_error_speed_slow=0.4,
            joint_braking_exponent=3.0,
            singularity_ratio_stop=0.01,
        )

        self.assertNotIn("--ik-profile", options)
        self.assertNotIn("--frame-error-speed-slow", options)
        self.assertNotIn("--joint-braking-exponent", options)
        self.assertNotIn("--singularity-ratio-stop", options)
        self.assertEqual(params.frame_error_speed_slow, 0.4)
        self.assertEqual(params.joint_braking_exponent, 3.0)
        self.assertEqual(params.singularity_ratio_stop, 0.01)

    def test_six_control_overrides_and_velocity_yaml(self) -> None:
        custom_caps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "velocity.yaml"
            path.write_text(
                "arm_velocity_limits:\n"
                + "".join(f"  - {value}\n" for value in custom_caps),
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            register_ik_args(parser)
            params = ik_params_from_args(
                parser.parse_args(
                    [
                        "--limit-velocity",
                        "--config",
                        str(path),
                        "--frame-position-error-limit",
                        "0.004",
                        "--nullspace-cost",
                        "8",
                        "--nullspace-return-rate",
                        "1.2",
                        "--joint-braking-distance",
                        "0.4",
                        "--singularity-max-approach-rate",
                        "0.2",
                        "--kinetic-energy-cost",
                        "0.00004",
                    ]
                )
            )

        self.assertEqual(params.frame_position_error_limit, 0.004)
        self.assertEqual(params.nullspace_cost, 8.0)
        self.assertEqual(params.nullspace_return_rate, 1.2)
        self.assertEqual(params.joint_braking_distance, 0.4)
        self.assertEqual(params.singularity_max_approach_rate, 0.2)
        self.assertEqual(params.kinetic_energy_cost, 4e-5)
        assert params.velocity_limits is not None
        for side in ("left", "right"):
            for index, expected in enumerate(custom_caps):
                self.assertEqual(
                    params.velocity_limits[f"openarm_{side}_joint{index + 1}"],
                    expected,
                )

    def test_diag_reg_is_not_a_registered_solver_option(self) -> None:
        parser = argparse.ArgumentParser()
        register_ik_args(parser)
        options = {
            option for action in parser._actions for option in action.option_strings
        }
        self.assertNotIn("--diag-reg", options)

    def test_braking_can_be_disabled_without_disabling_velocity_limits(self) -> None:
        parser = argparse.ArgumentParser()
        register_ik_args(parser)
        params = ik_params_from_args(
            parser.parse_args(["--limit-velocity", "--no-joint-braking"])
        )

        self.assertFalse(params.joint_braking)
        self.assertIsNotNone(params.velocity_limits)
        solver = Kinematics(_setup("right"), params)._ik
        assert solver is not None
        assert solver._joint_limit is not None
        self.assertIsNone(solver._joint_limit.braking_distance)


class RelativeFrameTest(unittest.TestCase):
    """Exercise upstream relative-frame handling through the wrapper."""

    def test_latest_model_contains_default_arm_origin(self) -> None:
        setup = ArmSetup.from_args(
            xml=openarm_mujoco.openarm_cell_xml(),
            mode="right",
            frame_right="right_ee_control_point",
            frame_type_right="site",
            frame_left="left_ee_control_point",
            frame_type_left="site",
        )
        self.assertGreaterEqual(setup.origin_id, 0)
        self.assertEqual(setup.read_ee_pose("right").shape, (7,))

    def test_vr_solver_wraps_native_relative_frame_task(self) -> None:
        setup = _setup(
            "right",
            origin_frame="openarm_right_base_link",
            origin_frame_type="body",
        )
        kinematics = Kinematics(
            setup,
            IKParams(
                position_cost=10.0,
                orientation_cost=1.0,
                lm_damping=0.02,
                damping=0.1,
                posture_cost=0.0,
                dt=0.004,
                max_iters=5,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        task = solver._tasks["right"]
        self.assertIsInstance(task, BoundedFrameTask)
        assert isinstance(task, BoundedFrameTask)
        self.assertIsInstance(task.frame_task, mink.RelativeFrameTask)
        self.assertEqual(task.root_name, "openarm_right_base_link")

        kinematics.set_target("right", setup.read_ee_pose("right"))
        self.assertIsNotNone(kinematics.solve())


class BoundedFrameTaskTest(unittest.TestCase):
    """Verify error modulation without changing Mink target semantics."""

    def _task(
        self,
    ) -> tuple[ArmSetup, mink.Configuration, mink.FrameTask, BoundedFrameTask]:
        setup = _setup("right")
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        native = mink.FrameTask(
            "right_ee_control_point",
            "site",
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=0.02,
        )
        return (
            setup,
            configuration,
            native,
            BoundedFrameTask(
                native,
                position_error_limit=0.003,
                control_dt=0.004,
                speed_slow=0.6,
                speed_fast=0.9,
                latch_multiplier=2.0,
            ),
        )

    def test_limits_request_but_preserves_full_error_and_target(self) -> None:
        setup, configuration, native, task = self._task()
        target = setup.read_ee_pose("right").astype(np.float64)
        target[0] += 0.1
        task.set_target(pose_to_se3(target))

        full_error = task.compute_full_error(configuration)
        limited_error = task.compute_limited_error(configuration)

        self.assertGreater(float(np.linalg.norm(full_error[:3])), 0.09)
        self.assertAlmostEqual(float(np.linalg.norm(limited_error[:3])), 0.003)
        np.testing.assert_array_equal(limited_error[3:], full_error[3:])
        self.assertIsNotNone(native.transform_target_to_world)

    def test_zero_activation_is_exactly_the_native_objective(self) -> None:
        setup, configuration, native, task = self._task()
        target = setup.read_ee_pose("right").astype(np.float64)
        target[:3] += [0.03, -0.02, 0.01]
        task.set_target(pose_to_se3(target))
        task.set_limit_activation(0.0)

        native_objective = native.compute_qp_objective(configuration)
        wrapped_objective = task.compute_qp_objective(configuration)

        np.testing.assert_array_equal(wrapped_objective.H, native_objective.H)
        np.testing.assert_array_equal(wrapped_objective.c, native_objective.c)

    def test_speed_schedule_is_instant_and_latches_accumulated_error(self) -> None:
        setup = _setup("right")
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=5,
                posture_cost=0.0,
            ),
        )
        pose = setup.read_ee_pose("right").astype(np.float64)
        kinematics.set_target("right", pose)
        medium = pose.copy()
        medium[0] += 0.003  # 0.75 m/s, midpoint of the 0.6 -> 0.9 window.
        kinematics.set_target("right", medium)

        solver = kinematics._ik
        assert solver is not None
        task = solver._tasks["right"]
        assert isinstance(task, BoundedFrameTask)
        self.assertAlmostEqual(task.limit_activation, 0.5)

        far = medium.copy()
        far[0] += 0.02
        kinematics.set_target("right", far)
        self.assertEqual(task.limit_activation, 1.0)
        kinematics.set_target("right", far)
        self.assertEqual(task.limit_activation, 1.0)


class ArmJointLimitTest(unittest.TestCase):
    """Exercise the merged position, velocity, and braking inequalities."""

    def _limit(
        self,
        *,
        braking_distance: float | None = 0.5,
    ) -> tuple[ArmJointLimit, mink.Configuration]:
        setup = _setup("right")
        limit = ArmJointLimit(
            setup.model,
            setup.joint_resolver.arm_qpos_indices("right"),
            _velocity_mapping("right"),
            position_gain=0.95,
            braking_distance=braking_distance,
            braking_exponent=2.0,
            braking_reaction_time=0.04,
            braking_distance_buffer=0.01,
        )
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        return limit, configuration

    def test_center_uses_physical_velocity_cap(self) -> None:
        limit, configuration = self._limit(braking_distance=None)
        row = 0
        q = configuration.q
        q[limit.qpos_indices[row]] = 0.5 * (limit.lower[row] + limit.upper[row])
        configuration.update(q=q)
        constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
        assert constraint.h is not None

        self.assertAlmostEqual(
            constraint.h[row],
            limit.max_velocity[row] * 0.01,
        )
        self.assertAlmostEqual(
            constraint.h[limit.indices.size + row],
            limit.max_velocity[row] * 0.01,
        )

    def test_configuration_limit_uses_selected_dofs_and_gain(self) -> None:
        setup = _setup("right")
        limit = ArmConfigurationLimit(
            setup.model,
            setup.joint_resolver.arm_qpos_indices("right"),
            gain=0.8,
        )

        np.testing.assert_array_equal(
            limit.indices,
            setup.joint_resolver.arm_dof_indices("right"),
        )
        self.assertEqual(limit.gain, 0.8)

    def test_half_braking_distance_allows_quarter_velocity(self) -> None:
        limit, configuration = self._limit()
        row = 0
        q = configuration.q
        q[limit.qpos_indices[row]] = limit.lower[row] + 0.25
        configuration.update(q=q)
        constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
        assert constraint.h is not None

        lower_h = constraint.h[limit.indices.size + row]
        self.assertAlmostEqual(
            lower_h,
            0.25 * limit.max_velocity[row] * 0.01,
        )

    def test_measured_q_dq_reduce_effective_distance(self) -> None:
        limit, configuration = self._limit()
        row = 0
        qpos_index = limit.qpos_indices[row]
        dof_index = limit.dof_indices[row]
        measured_q = configuration.q
        measured_q[qpos_index] = limit.lower[row] + 0.1
        measured_dq = np.zeros(configuration.model.nv)
        measured_dq[dof_index] = -1.0
        limit.update_measured_state(measured_q, measured_dq)
        limit.compute_qp_inequalities(configuration, dt=0.01)

        state = limit.last_state
        assert state is not None
        assert state.lower_distance is not None
        self.assertAlmostEqual(state.lower_distance[row], 0.05)
        self.assertGreater(state.lower_step[row], -1e-3)

    def test_overshoot_has_one_feasible_recovery_step(self) -> None:
        limit, configuration = self._limit()
        row = 0
        q = configuration.q
        q[limit.qpos_indices[row]] = limit.lower[row] - 0.1
        configuration.update(q=q)
        limit.compute_qp_inequalities(configuration, dt=0.01)
        state = limit.last_state
        assert state is not None

        self.assertGreater(state.lower_step[row], 0.0)
        self.assertAlmostEqual(state.lower_step[row], state.upper_step[row])
        self.assertLessEqual(
            state.upper_step[row],
            limit.max_velocity[row] * 0.01,
        )


class NullspaceTaskTest(unittest.TestCase):
    """Exercise exact one-dimensional nullspace home regularization."""

    def test_structural_direction_is_null_and_sign_continuous(self) -> None:
        rng = np.random.default_rng(7)
        jacobian = rng.normal(size=(6, 7))
        direction, singular_values = structural_nullspace_direction(jacobian)
        aligned, _ = structural_nullspace_direction(jacobian, previous=-direction)

        self.assertEqual(singular_values.shape, (6,))
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)
        self.assertLess(float(np.linalg.norm(jacobian @ direction)), 1e-12)
        self.assertGreater(float(aligned @ (-direction)), 1.0 - 1e-12)

    def test_smooth_activation_has_flat_clamped_endpoints(self) -> None:
        self.assertEqual(smoothstep_activation(0.02, 0.02, 0.05), 0.0)
        self.assertAlmostEqual(smoothstep_activation(0.035, 0.02, 0.05), 0.5)
        self.assertEqual(smoothstep_activation(0.05, 0.02, 0.05), 1.0)

    def test_task_is_nullspace_only_and_caps_return_speed(self) -> None:
        setup = _setup("right")
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        frame_task = mink.FrameTask(
            "right_ee_control_point",
            "site",
            position_cost=10.0,
            orientation_cost=1.0,
        )
        dofs = setup.joint_resolver.arm_dof_indices("right")
        task = NullspacePostureTask(
            model=setup.model,
            frame_task=frame_task,
            dof_indices=dofs,
            home_qpos=configuration.q,
            cost=12.0,
            dt=0.0008,
            return_rate=100.0,
            max_speed=1.0,
            singularity_low=0.0,
            singularity_high=1e-9,
            characteristic_length=0.3,
        )
        task.compute_qp_objective(configuration)
        initial = task.last_state
        assert initial is not None

        q = configuration.q
        tangent = np.zeros(setup.model.nv)
        tangent[dofs] = initial.direction
        mujoco.mj_integratePos(setup.model, q, tangent, 0.5)
        configuration.update(q=q)
        task.compute_qp_objective(configuration)
        state = task.last_state
        assert state is not None

        self.assertLess(state.jacobian_residual, 1e-10)
        self.assertLessEqual(abs(state.return_speed), 1.0)
        self.assertAlmostEqual(abs(state.displacement), 0.0008)

    def test_sync_does_not_move_home_reference(self) -> None:
        setup = _setup()
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=5,
                posture_cost=0.0,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        homes = {
            side: task._home_qpos.copy()
            for side, task in solver._nullspace_tasks.items()
        }
        kinematics.sync(np.linspace(-0.2, 0.2, 16, dtype=np.float32))
        for side, task in solver._nullspace_tasks.items():
            np.testing.assert_array_equal(task._home_qpos, homes[side])


class SingularityLimitTest(unittest.TestCase):
    """Exercise one-sided geometric singularity approach limiting."""

    def _limit(
        self,
    ) -> tuple[ArmSetup, SingularityApproachLimit, mink.Configuration]:
        setup = _setup("right")
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        task = mink.FrameTask(
            "right_ee_control_point",
            "site",
            position_cost=10.0,
            orientation_cost=1.0,
        )
        limit = SingularityApproachLimit(
            setup.model,
            task,
            setup.joint_resolver.arm_dof_indices("right"),
            characteristic_length=0.3,
            ratio_stop=0.02,
            ratio_slow=0.08,
            max_approach_rate=0.25,
            exponent=2.0,
        )
        return setup, limit, configuration

    def test_only_approaching_gradient_component_is_bounded(self) -> None:
        _, limit, configuration = self._limit()
        limit.prepare(configuration)
        constraint = limit.compute_qp_inequalities(configuration, dt=0.004)
        state = limit.last_state
        assert state is not None
        assert constraint.G is not None
        approach = np.zeros(configuration.model.nv)
        approach[limit.dof_indices] = -state.gradient
        self.assertGreater(float((constraint.G @ approach)[0]), 0.0)
        self.assertLess(float((constraint.G @ (-approach))[0]), 0.0)

    def test_target_does_not_change_geometric_ratio(self) -> None:
        setup, limit, configuration = self._limit()
        limit.prepare(configuration)
        initial = limit.last_state
        assert initial is not None

        wrapped = BoundedFrameTask(
            mink.FrameTask(
                "right_ee_control_point",
                "site",
                position_cost=10.0,
                orientation_cost=1.0,
            ),
            position_error_limit=0.003,
            control_dt=0.004,
            speed_slow=0.6,
            speed_fast=0.9,
            latch_multiplier=2.0,
        )
        target = setup.read_ee_pose("right").astype(np.float64)
        target[:3] += [0.2, -0.1, 0.15]
        wrapped.set_target(pose_to_se3(target))
        second = SingularityApproachLimit(
            setup.model,
            wrapped,
            limit.dof_indices,
            characteristic_length=0.3,
            ratio_stop=0.02,
            ratio_slow=0.08,
            max_approach_rate=0.25,
        )
        second.prepare(configuration)
        state = second.last_state
        assert state is not None

        self.assertAlmostEqual(state.command_ratio, initial.command_ratio)
        np.testing.assert_allclose(state.gradient, initial.gradient, atol=1e-12)


class SolverTest(unittest.TestCase):
    """Exercise solver wiring, timing, state handling, and failure recovery."""

    def test_flat_params_wire_control_features_and_full_posture(self) -> None:
        setup = _setup("right")
        params = IKParams(
            posture_cost=0.01,
            dt=0.004,
            velocity_limits=_velocity_mapping("right"),
        )
        kinematics = Kinematics(
            setup,
            params,
        )
        solver = kinematics._ik
        assert solver is not None

        self.assertEqual(solver._posture_cost, 0.01)
        assert solver._joint_limit is not None
        self.assertEqual(solver._joint_limit.braking_distance, 0.5)
        self.assertIsInstance(solver._tasks["right"], BoundedFrameTask)
        self.assertIsInstance(solver._joint_limit, ArmJointLimit)
        assert solver._joint_limit is not None
        self.assertEqual(
            solver._joint_limit.braking_distance,
            params.joint_braking_distance,
        )
        self.assertIn("right", solver._nullspace_tasks)
        self.assertIn("right", solver._singularity_limits)
        self.assertIsNotNone(solver._kinetic_energy_task)

        kinematics.set_target("right", setup.read_ee_pose("right"))
        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            return_value=np.zeros(setup.model.nv),
        ) as solve:
            self.assertIsNotNone(kinematics.solve())
        self.assertIn(solver._posture_task, solve.call_args.args[1])

    def test_substeps_cover_one_control_period_in_physical_units(self) -> None:
        setup = _setup()
        velocity_limits = {
            f"openarm_{side}_joint{index + 1}": float(cap)
            for side in ("left", "right")
            for index, cap in enumerate(np.linspace(0.2, 0.8, 7))
        }
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=10,
                damping=0.25,
                posture_cost=0.0,
                velocity_limits=velocity_limits,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        self.assertEqual(solver._substep_dt, 0.0004)
        self.assertIsInstance(solver._limits[0], ArmJointLimit)

        velocity = np.zeros(setup.model.nv)
        for side in ("right", "left"):
            velocity[setup.joint_resolver.arm_dof_indices(side)] = np.linspace(
                0.2, 0.8, 7
            )
        q_before = solver._config.q.copy()
        for side in setup.sides:
            kinematics.set_target(side, setup.read_ee_pose(side))

        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            return_value=velocity,
        ) as solve:
            result = kinematics.solve()

        self.assertIsNotNone(result)
        self.assertEqual(solve.call_count, 10)
        for call in solve.call_args_list:
            self.assertEqual(call.args[2], 0.0004)
            self.assertEqual(call.kwargs["damping"], 0.25)
            self.assertNotIn("diag_reg", call.kwargs)
        np.testing.assert_allclose(
            solver._config.q - q_before,
            velocity * 0.004,
            atol=1e-12,
        )

    def test_measured_state_does_not_sync_command_and_can_be_cleared(self) -> None:
        setup = _setup()
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                velocity_limits=_velocity_mapping("right", "left"),
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        command_before = solver._config.q.copy()
        measured = _driver_state(setup)
        measured[0] += 0.1

        kinematics.update_measured_state(measured, np.zeros(16))

        np.testing.assert_array_equal(solver._config.q, command_before)
        assert solver._joint_limit is not None
        self.assertIsNotNone(solver._joint_limit._measured_qpos)
        kinematics.clear_measured_state()
        self.assertIsNone(solver._joint_limit._measured_qpos)

    def test_sync_does_not_overwrite_gripper_command(self) -> None:
        setup = _setup()
        kinematics = Kinematics(setup, IKParams())
        kinematics.set_gripper("right", 0.7)
        state = _driver_state(setup)
        state[7] = 0.1
        kinematics.sync(state)
        solver = kinematics._ik
        assert solver is not None
        self.assertAlmostEqual(float(solver._gripper[0]), 0.7)

    def test_failure_rolls_back_and_requires_fresh_targets(self) -> None:
        setup = _setup()
        kinematics = Kinematics(
            setup,
            IKParams(dt=0.004, max_iters=3, posture_cost=0.0),
        )
        solver = kinematics._ik
        assert solver is not None
        for side in setup.sides:
            kinematics.set_target(side, setup.read_ee_pose(side))
        q_before = solver._config.q.copy()
        velocity = np.zeros(setup.model.nv)
        velocity[solver._arm_dofs_by_side["right"][0]] = 0.5

        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            side_effect=[velocity, mink.exceptions.NoSolutionFound("daqp")],
        ):
            result = kinematics.solve()

        self.assertIsNone(result)
        np.testing.assert_array_equal(solver._config.q, q_before)
        self.assertFalse(kinematics.ready())
        kinematics.set_target("right", setup.read_ee_pose("right"))
        self.assertFalse(kinematics.ready())
        kinematics.set_target("left", setup.read_ee_pose("left"))
        self.assertTrue(kinematics.ready())

    def test_right_left_and_bimanual_real_qp_solve(self) -> None:
        for mode in ("right", "left", "bimanual"):
            with self.subTest(mode=mode):
                setup = _setup(mode, origin_frame="arm_origin")
                kinematics = Kinematics(
                    setup,
                    IKParams(
                        position_cost=10.0,
                        orientation_cost=1.0,
                        lm_damping=0.02,
                        damping=0.1,
                        posture_cost=0.0,
                        dt=0.004,
                        max_iters=5,
                        velocity_limits=_velocity_mapping(*setup.sides),
                    ),
                )
                for side in setup.sides:
                    kinematics.set_target(side, setup.read_ee_pose(side))
                result = kinematics.solve()
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.shape, (16,))

    def test_single_arm_mode_freezes_inactive_arm_dofs(self) -> None:
        setup = _setup("right")
        kinematics = Kinematics(setup, IKParams())
        solver = kinematics._ik
        assert solver is not None
        assert solver._freeze_task is not None

        frozen = set(solver._freeze_task.dof_indices)
        right = set(setup.joint_resolver.arm_dof_indices("right"))
        left = set(setup.joint_resolver.arm_dof_indices("left"))
        self.assertTrue(left <= frozen)
        self.assertTrue(right.isdisjoint(frozen))

    def test_kinetic_energy_uses_current_mujoco_inertia_api(self) -> None:
        setup = _setup("right")
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                kinetic_energy_cost=1e-7,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        task = solver._kinetic_energy_task
        assert task is not None
        objective = task.compute_qp_objective(solver._config)

        self.assertEqual(objective.H.shape, (setup.model.nv, setup.model.nv))
        self.assertTrue(np.all(np.isfinite(objective.H)))
        np.testing.assert_allclose(objective.H, objective.H.T, atol=1e-12)
        np.testing.assert_array_equal(objective.c, np.zeros(setup.model.nv))


if __name__ == "__main__":
    unittest.main()
