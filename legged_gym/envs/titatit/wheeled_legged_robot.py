"""Wheeled-legged robot environment for WMP framework.

Handles the key differences from A1 quadruped:
- 16 DOF (12 leg + 4 wheel joints at indices 3,7,11,15)
- Wheel joints use velocity/torque control instead of PD position control
- Action reindex between SDK order and Isaac order
- Low-pass action filter
- Hip scale reduction
- Forward heightmap (525 points) for depth prediction
- Titatit-specific rewards (terrain_progress, corridor, lateral_deviation)
"""

import numpy as np
import random

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.math import quat_apply_yaw


# SDK/action order matches the Isaac Gym DOF order loaded from the Titatit URDF:
# FL_hip, FL_thigh, FL_calf, FL_foot,
# FR_hip, FR_thigh, FR_calf, FR_foot,
# RL_hip, RL_thigh, RL_calf, RL_foot,
# RR_hip, RR_thigh, RR_calf, RR_foot.
REINDEX_SDK_TO_ISAAC = list(range(16))
REINDEX_ISAAC_TO_SDK = list(range(16))

# Wheel joint indices in Isaac order
WHEEL_INDICES_ISAAC = [3, 7, 11, 15]
# Hip joint indices in Isaac order
HIP_INDICES_ISAAC = [0, 4, 8, 12]


class WheeledLeggedRobot(LeggedRobot):
    """Titatit wheeled-legged robot environment.

    Inherits from LeggedRobot (A1 base) and overrides:
    - step(): action reindex + low-pass filter
    - _compute_torques(): wheel-specific torque control
    - compute_observations(): 16 DOF observation structure
    - _get_noise_scale_vec(): 16 DOF noise structure
    - Various reward functions: Titatit-specific rewards
    """

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Pre-compute reindex tensors
        self.reindex_sdk_to_isaac = torch.tensor(REINDEX_SDK_TO_ISAAC, device=self.device, dtype=torch.long)
        self.reindex_isaac_to_sdk = torch.tensor(REINDEX_ISAAC_TO_SDK, device=self.device, dtype=torch.long)

        # Wheel joint indices (in Isaac order: FL, FR, RL, RR)
        self.wheel_indices = torch.tensor(WHEEL_INDICES_ISAAC, device=self.device, dtype=torch.long)
        # Hip joint indices (in Isaac order: FL, FR, RL, RR)
        self.hip_indices = torch.tensor(HIP_INDICES_ISAAC, device=self.device, dtype=torch.long)
        self.leg_indices = torch.tensor(
            [i for i in range(self.num_dof) if i not in WHEEL_INDICES_ISAAC],
            device=self.device,
            dtype=torch.long,
        )
        self.rear_leg_indices = torch.tensor([8, 9, 10, 12, 13, 14], device=self.device, dtype=torch.long)
        self.front_foot_indices = self._find_body_indices(["FL_foot", "FR_foot"])
        self.rear_foot_indices = self._find_body_indices(["RL_foot", "RR_foot"])
        self.obstacle_foot_indices = self._find_body_indices(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
        self.obstacle_leg_joint_indices = torch.tensor(
            [[0, 1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14]],
            device=self.device,
            dtype=torch.long,
        )
        self._validate_dof_order()

        # Wheel PD gains (different from leg joints)
        self.wheel_kp = getattr(self.cfg.control, 'wheel_kp', 10.0)
        self.wheel_kd_scale = getattr(self.cfg.control, 'wheel_kd_scale', 0.5)

        # Hip scale reduction
        self.hip_scale_reduction = getattr(self.cfg.control, 'hip_scale_reduction', 0.5)

        # Low-pass filter flag
        self.use_filter = getattr(self.cfg.control, 'use_filter', True)

        # Terrain progress tracking
        self.terrain_progress_x0 = getattr(self.cfg.rewards, 'terrain_progress_x0', 2)
        self.terrain_progress_x1 = getattr(self.cfg.rewards, 'terrain_progress_x1', 12)
        self.terrain_progress_y_radius = getattr(self.cfg.rewards, 'terrain_progress_y_radius', 4)
        self.terrain_progress_trigger_height = getattr(self.cfg.rewards, 'terrain_progress_trigger_height', 0.04)
        self.terrain_progress_max_vel = getattr(self.cfg.rewards, 'terrain_progress_max_vel', 0.7)
        self.terrain_progress_impact_width = getattr(self.cfg.rewards, 'terrain_progress_impact_width', 0.6)
        self.terrain_progress_clearance_width = getattr(self.cfg.rewards, 'terrain_progress_clearance_width', 0.6)
        self.lateral_deviation_deadband = getattr(self.cfg.rewards, 'lateral_deviation_deadband', 0.25)
        self.terrain_corridor_deadband = getattr(self.cfg.rewards, 'terrain_corridor_deadband', 0.35)
        self.terrain_corridor_heading_width = getattr(self.cfg.rewards, 'terrain_corridor_heading_width', 0.7)
        self.gait_contact_force_threshold = getattr(self.cfg.rewards, 'gait_contact_force_threshold', 1.0)
        self.front_wheel_impact_force_threshold = getattr(self.cfg.rewards, 'front_wheel_impact_force_threshold', 30.0)
        self.front_wheel_impact_force_scale = getattr(self.cfg.rewards, 'front_wheel_impact_force_scale', 35.0)
        self.front_wheel_impact_vertical_ratio = getattr(self.cfg.rewards, 'front_wheel_impact_vertical_ratio', 0.6)
        self.front_wheel_impact_ratio_scale = getattr(self.cfg.rewards, 'front_wheel_impact_ratio_scale', 0.8)
        self.front_wheel_impact_speed_threshold = getattr(self.cfg.rewards, 'front_wheel_impact_speed_threshold', 0.35)
        self.front_wheel_impact_speed_scale = getattr(self.cfg.rewards, 'front_wheel_impact_speed_scale', 0.35)
        self.front_wheel_impact_max_penalty = getattr(self.cfg.rewards, 'front_wheel_impact_max_penalty', 1.5)
        self.front_clearance_light_contact_force = getattr(self.cfg.rewards, 'front_clearance_light_contact_force', 8.0)
        self.front_clearance_horizontal_force = getattr(self.cfg.rewards, 'front_clearance_horizontal_force', 12.0)
        self.controlled_front_contact_min_force = getattr(self.cfg.rewards, 'controlled_front_contact_min_force', 5.0)
        self.controlled_front_contact_max_force = getattr(self.cfg.rewards, 'controlled_front_contact_max_force', 55.0)
        self.controlled_front_contact_max_ratio = getattr(self.cfg.rewards, 'controlled_front_contact_max_ratio', 1.2)
        self.controlled_front_contact_speed_target = getattr(self.cfg.rewards, 'controlled_front_contact_speed_target', 0.22)
        self.controlled_front_contact_speed_width = getattr(self.cfg.rewards, 'controlled_front_contact_speed_width', 0.22)
        self.front_touchdown_impact_force_threshold = getattr(
            self.cfg.rewards, 'front_touchdown_impact_force_threshold', 70.0
        )
        self.front_touchdown_impact_force_scale = getattr(
            self.cfg.rewards, 'front_touchdown_impact_force_scale', 90.0
        )
        self.front_touchdown_impact_down_speed_threshold = getattr(
            self.cfg.rewards, 'front_touchdown_impact_down_speed_threshold', 0.15
        )
        self.front_touchdown_impact_down_speed_scale = getattr(
            self.cfg.rewards, 'front_touchdown_impact_down_speed_scale', 0.45
        )
        self.early_front_clearance_x0 = getattr(self.cfg.rewards, 'early_front_clearance_x0', 6)
        self.early_front_clearance_x1 = getattr(self.cfg.rewards, 'early_front_clearance_x1', 18)
        self.early_front_clearance_trigger_height = getattr(self.cfg.rewards, 'early_front_clearance_trigger_height', 0.04)
        self.early_front_clearance_impact_threshold = getattr(self.cfg.rewards, 'early_front_clearance_impact_threshold', 15.0)
        self.obstacle_front_air_trigger_height = getattr(
            self.cfg.rewards, 'obstacle_front_air_trigger_height', self.early_front_clearance_trigger_height
        )
        self.obstacle_front_air_preimpact_force = getattr(
            self.cfg.rewards, 'obstacle_front_air_preimpact_force', self.front_clearance_horizontal_force
        )
        self.obstacle_front_air_preimpact_score = getattr(
            self.cfg.rewards, 'obstacle_front_air_preimpact_score', 0.05
        )
        self.obstacle_front_clearance_margin = getattr(self.cfg.rewards, 'obstacle_front_clearance_margin', 0.04)
        self.obstacle_front_clearance_min_target = getattr(self.cfg.rewards, 'obstacle_front_clearance_min_target', 0.08)
        self.obstacle_front_clearance_max_target = getattr(self.cfg.rewards, 'obstacle_front_clearance_max_target', 0.18)
        self.obstacle_front_clearance_far_distance = getattr(
            self.cfg.rewards, 'obstacle_front_clearance_far_distance', 1.6
        )
        self.obstacle_front_clearance_deadline_distance = getattr(
            self.cfg.rewards, 'obstacle_front_clearance_deadline_distance', 0.75
        )
        self.drop_front_clearance_margin = getattr(self.cfg.rewards, 'drop_front_clearance_margin', 0.03)
        self.drop_front_clearance_min_target = getattr(self.cfg.rewards, 'drop_front_clearance_min_target', 0.10)
        self.drop_front_clearance_max_target = getattr(self.cfg.rewards, 'drop_front_clearance_max_target', 0.18)
        self.gap_tracking_lin_vel_x = getattr(self.cfg.rewards, 'gap_tracking_lin_vel_x', 0.28)
        self.gap_front_clearance_margin = getattr(self.cfg.rewards, 'gap_front_clearance_margin', 0.03)
        self.gap_front_clearance_min_target = getattr(self.cfg.rewards, 'gap_front_clearance_min_target', 0.18)
        self.gap_front_clearance_max_target = getattr(self.cfg.rewards, 'gap_front_clearance_max_target', 0.24)
        self.front_stagger_clearance_threshold = getattr(self.cfg.rewards, 'front_stagger_clearance_threshold', 0.04)
        self.front_swing_clearance_target = getattr(self.cfg.rewards, 'front_swing_clearance_target', 0.10)
        self.obstacle_front_lift_target = getattr(self.cfg.rewards, 'obstacle_front_lift_target', 0.18)
        self.front_swing_stability_width = getattr(self.cfg.rewards, 'front_swing_stability_width', 0.35)
        self.obstacle_front_swing_start_distance = getattr(
            self.cfg.rewards, 'obstacle_front_swing_start_distance', 1.45
        )
        self.obstacle_front_swing_peak_distance = getattr(
            self.cfg.rewards, 'obstacle_front_swing_peak_distance', 0.95
        )
        self.obstacle_front_swing_end_distance = getattr(
            self.cfg.rewards, 'obstacle_front_swing_end_distance', 0.45
        )
        self.obstacle_front_swing_clearance_margin = getattr(
            self.cfg.rewards, 'obstacle_front_swing_clearance_margin', 0.03
        )
        self.obstacle_front_swing_min_clearance = getattr(
            self.cfg.rewards, 'obstacle_front_swing_min_clearance', 0.08
        )
        self.obstacle_front_swing_min_lift = getattr(
            self.cfg.rewards, 'obstacle_front_swing_min_lift', 0.015
        )
        self.obstacle_front_swing_max_clearance = getattr(
            self.cfg.rewards, 'obstacle_front_swing_max_clearance', 0.18
        )
        self.obstacle_front_swing_impact_threshold = getattr(
            self.cfg.rewards, 'obstacle_front_swing_impact_threshold', 8.0
        )
        self.obstacle_front_air_min_time = getattr(self.cfg.rewards, 'obstacle_front_air_min_time', 0.08)
        self.obstacle_front_air_stair_target_time = getattr(
            self.cfg.rewards, 'obstacle_front_air_stair_target_time', 0.16
        )
        self.obstacle_front_air_gap_target_time = getattr(
            self.cfg.rewards, 'obstacle_front_air_gap_target_time', 0.22
        )
        self.obstacle_front_air_pit_target_time = getattr(
            self.cfg.rewards, 'obstacle_front_air_pit_target_time', 0.14
        )
        self.obstacle_front_air_max_time = getattr(self.cfg.rewards, 'obstacle_front_air_max_time', 0.36)
        self.obstacle_front_air_gap_max_time = getattr(
            self.cfg.rewards, 'obstacle_front_air_gap_max_time', 0.45
        )
        self.obstacle_front_air_landing_force = getattr(
            self.cfg.rewards, 'obstacle_front_air_landing_force', 10.0
        )
        self.obstacle_front_air_lateral_speed = getattr(
            self.cfg.rewards, 'obstacle_front_air_lateral_speed', 0.35
        )
        self.obstacle_front_air_down_speed = getattr(
            self.cfg.rewards, 'obstacle_front_air_down_speed', 0.65
        )
        self.obstacle_front_air_progress_vel = getattr(
            self.cfg.rewards, 'obstacle_front_air_progress_vel', 0.35
        )
        self.obstacle_dead_leg_front_air_time = getattr(
            self.cfg.rewards, 'obstacle_dead_leg_front_air_time', 0.36
        )
        self.obstacle_dead_leg_rear_air_time = getattr(
            self.cfg.rewards, 'obstacle_dead_leg_rear_air_time', 0.20
        )
        self.obstacle_dead_leg_clearance = getattr(
            self.cfg.rewards, 'obstacle_dead_leg_clearance', 0.08
        )
        self.obstacle_dead_leg_clearance_width = getattr(
            self.cfg.rewards, 'obstacle_dead_leg_clearance_width', 0.08
        )
        self.obstacle_dead_leg_dof_error = getattr(
            self.cfg.rewards, 'obstacle_dead_leg_dof_error', 0.10
        )
        self.obstacle_dead_leg_dof_width = getattr(
            self.cfg.rewards, 'obstacle_dead_leg_dof_width', 0.25
        )
        self.obstacle_dead_leg_min_progress_vel = getattr(
            self.cfg.rewards, 'obstacle_dead_leg_min_progress_vel', 0.12
        )
        self.obstacle_posture_limit = getattr(
            self.cfg.rewards, 'obstacle_posture_limit', 0.24
        )
        self.obstacle_posture_width = getattr(
            self.cfg.rewards, 'obstacle_posture_width', 0.24
        )
        self.obstacle_ang_vel_xy_limit = getattr(
            self.cfg.rewards, 'obstacle_ang_vel_xy_limit', 0.50
        )
        self.obstacle_ang_vel_xy_width = getattr(
            self.cfg.rewards, 'obstacle_ang_vel_xy_width', 1.00
        )
        self.obstacle_front_air_stair_weight = getattr(
            self.cfg.rewards, 'obstacle_front_air_stair_weight', 1.0
        )
        self.obstacle_front_air_gap_weight = getattr(
            self.cfg.rewards, 'obstacle_front_air_gap_weight', 0.9
        )
        self.obstacle_front_air_pit_weight = getattr(
            self.cfg.rewards, 'obstacle_front_air_pit_weight', 0.55
        )
        self.obstacle_front_edge_prepare_distance = getattr(
            self.cfg.rewards, 'obstacle_front_edge_prepare_distance', 1.35
        )
        self.obstacle_front_edge_deadline_distance = getattr(
            self.cfg.rewards, 'obstacle_front_edge_deadline_distance', 0.70
        )
        self.obstacle_front_edge_close_distance = getattr(
            self.cfg.rewards, 'obstacle_front_edge_close_distance', 0.30
        )
        self.obstacle_front_edge_clean_force = getattr(
            self.cfg.rewards, 'obstacle_front_edge_clean_force', 8.0
        )
        self.obstacle_front_edge_progress_vel = getattr(
            self.cfg.rewards, 'obstacle_front_edge_progress_vel', 0.45
        )
        self.obstacle_front_edge_stuck_speed = getattr(
            self.cfg.rewards, 'obstacle_front_edge_stuck_speed', 0.10
        )
        self.no_obstacle_front_air_clearance = getattr(
            self.cfg.rewards, 'no_obstacle_front_air_clearance', 0.03
        )
        self.no_obstacle_front_air_contact_force = getattr(
            self.cfg.rewards, 'no_obstacle_front_air_contact_force', self.gait_contact_force_threshold
        )
        self.no_obstacle_front_air_speed_threshold = getattr(
            self.cfg.rewards, 'no_obstacle_front_air_speed_threshold', 0.1
        )
        self.no_obstacle_front_air_height_width = getattr(
            self.cfg.rewards, 'no_obstacle_front_air_height_width', 0.05
        )
        self.obstacle_base_height_target = getattr(self.cfg.rewards, 'obstacle_base_height_target', 0.48)
        self.obstacle_base_height_margin = getattr(self.cfg.rewards, 'obstacle_base_height_margin', 0.10)
        self.base_height_low_target = getattr(self.cfg.rewards, 'base_height_low_target', 0.40)
        self.base_height_low_margin = getattr(self.cfg.rewards, 'base_height_low_margin', 0.10)
        self.obstacle_tracking_lin_vel_x = getattr(self.cfg.rewards, 'obstacle_tracking_lin_vel_x', 0.35)
        self.stair_front_clearance_deadline_distance = getattr(
            self.cfg.rewards, 'stair_front_clearance_deadline_distance', 1.05
        )
        self.stair_front_clearance_close_distance = getattr(
            self.cfg.rewards, 'stair_front_clearance_close_distance', 0.55
        )
        self.pit_front_wheel_impact_weight = getattr(self.cfg.rewards, 'pit_front_wheel_impact_weight', 0.45)
        self.pit_progress_impact_weight = getattr(self.cfg.rewards, 'pit_progress_impact_weight', 0.25)
        self.pit_stuck_multiplier = getattr(self.cfg.rewards, 'pit_stuck_multiplier', 1.6)
        self.forward_height_x_values = torch.tensor(
            self.cfg.terrain.measured_forward_points_x,
            device=self.device,
            dtype=torch.float,
        )
        self.stairup_env_mask = self._build_env_mask([
            (self.stairup_start_idx, self.stairup_end_idx),
        ])
        self.stairdown_env_mask = self._build_env_mask([
            (self.stairdown_start_idx, self.stairdown_end_idx),
        ])
        self.pit_env_mask = self._build_env_mask([
            (self.pit_start_idx, self.pit_end_idx),
        ])
        self.gap_env_mask = self._build_env_mask([
            (self.gap_start_idx, self.gap_end_idx),
        ])
        self.high_obstacle_env_mask = self.stairup_env_mask | self.pit_env_mask
        self.drop_obstacle_env_mask = self.stairdown_env_mask | self.gap_env_mask
        self.stagger_obstacle_env_mask = self.high_obstacle_env_mask
        self.step_obstacle_env_mask = self.high_obstacle_env_mask | self.drop_obstacle_env_mask
        self.front_obstacle_air_env_mask = self.stairup_env_mask | self.gap_env_mask | self.pit_env_mask
        self.front_obstacle_air_time = torch.zeros(
            self.num_envs,
            self.front_foot_indices.numel(),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.front_obstacle_air_max_clearance = torch.zeros_like(self.front_obstacle_air_time)
        self.obstacle_foot_air_time = torch.zeros(
            self.num_envs,
            self.obstacle_foot_indices.numel(),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.front_obstacle_air_active = torch.zeros(
            self.num_envs,
            self.front_foot_indices.numel(),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.front_obstacle_air_blocked = torch.zeros_like(self.front_obstacle_air_active)
        self.last_front_obstacle_contacts = torch.zeros(
            self.num_envs,
            self.front_foot_indices.numel(),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

    def _validate_dof_order(self):
        expected = {
            0: "FL_hip_joint",
            1: "FL_thigh_joint",
            2: "FL_calf_joint",
            3: "FL_foot_joint",
            4: "FR_hip_joint",
            5: "FR_thigh_joint",
            6: "FR_calf_joint",
            7: "FR_foot_joint",
            8: "RL_hip_joint",
            9: "RL_thigh_joint",
            10: "RL_calf_joint",
            11: "RL_foot_joint",
            12: "RR_hip_joint",
            13: "RR_thigh_joint",
            14: "RR_calf_joint",
            15: "RR_foot_joint",
        }
        mismatches = [
            f"{idx}: expected {name}, got {self.dof_names[idx] if idx < len(self.dof_names) else '<missing>'}"
            for idx, name in expected.items()
            if idx >= len(self.dof_names) or self.dof_names[idx] != name
        ]
        if mismatches:
            raise RuntimeError(
                "Titatit DOF order does not match the action/reward mapping:\n"
                + "\n".join(mismatches)
            )

    def _find_body_indices(self, body_names):
        body_indices = []
        for name in body_names:
            body_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
            if body_idx < 0:
                raise RuntimeError(f"Could not find rigid body '{name}' in Titatit asset")
            body_indices.append(body_idx)
        return torch.tensor(body_indices, device=self.device, dtype=torch.long)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, 'front_obstacle_air_time'):
            self.front_obstacle_air_time[env_ids] = 0.0
            self.front_obstacle_air_max_clearance[env_ids] = 0.0
            self.obstacle_foot_air_time[env_ids] = 0.0
            self.front_obstacle_air_active[env_ids] = False
            self.front_obstacle_air_blocked[env_ids] = False
            self.last_front_obstacle_contacts[env_ids] = False

    def _build_env_mask(self, ranges):
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for start_idx, end_idx in ranges:
            if end_idx > start_idx:
                mask[start_idx:end_idx] = True
        return mask

    def reindex(self, tensor):
        """Reindex from SDK joint order to Isaac joint order (16 DOF)."""
        return tensor[:, REINDEX_SDK_TO_ISAAC]

    def reindex_inverse(self, tensor):
        """Reindex from Isaac joint order to SDK joint order (16 DOF)."""
        return tensor[:, REINDEX_ISAAC_TO_SDK]

    def step(self, actions):
        """Apply actions with reindex and low-pass filter, then simulate.

        Args:
            actions: (num_envs, 16) in SDK joint order
        Returns:
            policy_obs, privileged_obs, rewards, reset_buf, extras, reset_env_ids, terminal_amp_states
        """
        self.global_counter += 1
        self.total_env_steps_counter += 1

        clip_actions = self.cfg.normalization.clip_actions
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # Low-pass action filter (in SDK order)
        # self.last_actions is in Isaac order (from previous step), so reindex back to SDK first
        if self.use_filter:
            last_actions_sdk = self.reindex_inverse(self.last_actions)
            actions = last_actions_sdk * 0.2 + actions * 0.8

        # Reindex actions from SDK order to Isaac order
        actions_isaac = self.reindex(actions)

        # Hip scale reduction (in Isaac order, hips are at indices 0,4,8,12)
        actions_isaac[:, self.hip_indices] *= self.hip_scale_reduction

        self.actions = actions_isaac

        # Action latency randomization
        rng = self.latency_range
        action_latency = random.randint(rng[0], rng[1])

        # Step physics
        self.render()
        for _ in range(self.cfg.control.decimation):
            if self.cfg.domain_rand.randomize_action_latency and _ < action_latency:
                self.torques = self._compute_torques(self.last_actions).view(self.torques.shape)
            else:
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            if self.cfg.domain_rand.randomize_motor_strength:
                rng = self.cfg.domain_rand.motor_strength_range
                self.torques = self.torques * torch_rand_float(rng[0], rng[1], self.torques.shape, device=self.device)

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        reset_env_ids, terminal_amp_states = self.post_physics_step()

        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # Depth buffer
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]
        else:
            self.extras["depth"] = None

        return policy_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states

    def _compute_torques(self, actions):
        """Compute torques with separate control for leg joints and wheel joints.

        Leg joints: PD position control
            torques = Kp * (action_scale * action + default_pos - current_pos) - Kd * dof_vel
        Wheel joints [3,7,11,15]: Velocity/torque control (matches original Titatit)
            torques = kp_factor * 10 * action_scaled - 0.5 * kd_factor * dof_vel

        Args:
            actions: (num_envs, 16) in Isaac joint order, already filtered
        Returns:
            torques: (num_envs, 16)
        """
        actions_scaled = actions * self.cfg.control.action_scale

        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        # PD position control for leg joints
        torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel

        # Override wheel joints with velocity/torque control
        # Original Titatif formula: kp_factor * 10 * action - 0.5 * kd_factor * vel
        # Here p_gains/d_gains already include randomization (kp_factor/kd_factor)
        wheel_idx = self.wheel_indices
        torques[:, wheel_idx] = (
            p_gains[:, wheel_idx] * self.wheel_kp * actions_scaled[:, wheel_idx]
            - self.wheel_kd_scale * d_gains[:, wheel_idx] * self.dof_vel[:, wheel_idx]
        )

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):
        """Compute observations for 16 DOF wheeled-legged robot.

        Observation buffer layout (exact quadruped WMP template extension):
            [contact_flag(8), contact_force(12), domain_rand(38), base_lin_vel(3),
             ang_vel(3), gravity(3), cmd(3), dof_pos(16), dof_vel(16),
             actions(16), heights(187)]

        Total: 61 + 41 + 16 + 187 = 305
        obs_buf = privileged_obs_buf (actor uses history/cmd/wm_latent, not raw obs)

        WMP slicing:
            WM prop:  obs[:, 61 : 102]  (41 dims, excludes actions/base_lin_vel)
            command:  obs[:, 67 : 70]
            history:  obs[:, 61:67] + obs[:, 70:118] (54 dims/step)
        """
        # Build observations in Isaac joint order. base_lin_vel is first and is
        # covered by privileged_dim, matching the original quadruped template.
        dof_pos_obs = self.dof_pos - self.default_dof_pos
        dof_pos_obs[:, self.wheel_indices] = 0.0
        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,                    # 3, privileged-only
            self.base_ang_vel * self.obs_scales.ang_vel,                    # 3
            self.projected_gravity,                                         # 3
            self.commands[:, :3] * self.commands_scale,                     # 3
            dof_pos_obs * self.obs_scales.dof_pos,                          # 16
            self.dof_vel * self.obs_scales.dof_vel,                         # 16
            self.actions,                                                   # 16
        ), dim=-1)  # total: 60

        # Add heightmap (before domain_rand, matching base class)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_heights,
                -1, 1.
            ) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # Prepend domain randomization (matching base class: each is prepended)
        if self.cfg.domain_rand.randomize_friction:
            self.privileged_obs_buf = torch.cat((self.randomized_frictions, self.privileged_obs_buf), dim=-1)
        if self.cfg.domain_rand.randomize_restitution:
            self.privileged_obs_buf = torch.cat((self.randomized_restitutions, self.privileged_obs_buf), dim=-1)
        if self.cfg.domain_rand.randomize_base_mass:
            self.privileged_obs_buf = torch.cat((self.randomized_added_masses, self.privileged_obs_buf), dim=-1)
        if self.cfg.domain_rand.randomize_com_pos:
            self.privileged_obs_buf = torch.cat((self.randomized_com_pos * self.obs_scales.com_pos, self.privileged_obs_buf), dim=-1)
        if self.cfg.domain_rand.randomize_gains:
            self.privileged_obs_buf = torch.cat((
                (self.randomized_p_gains / self.p_gains - 1) * self.obs_scales.pd_gains,
                self.privileged_obs_buf
            ), dim=-1)
            self.privileged_obs_buf = torch.cat((
                (self.randomized_d_gains / self.d_gains - 1) * self.obs_scales.pd_gains,
                self.privileged_obs_buf
            ), dim=-1)

        # Prepend contact information (matching base class)
        contact_force = self.sensor_forces.flatten(1) * self.obs_scales.contact_force
        self.privileged_obs_buf = torch.cat((contact_force, self.privileged_obs_buf), dim=-1)
        contact_flag = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1
        self.privileged_obs_buf = torch.cat((contact_flag.float(), self.privileged_obs_buf), dim=-1)

        # Add noise
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec

        # obs_buf = privileged_obs_buf (num_obs == num_privileged_obs)
        # The actor only uses history, command, and wm_latent - not raw obs_buf
        self.obs_buf = torch.clone(self.privileged_obs_buf)

    def _get_noise_scale_vec(self, cfg):
        """Set noise scale vector for 16 DOF wheeled-legged robot.

        Structure matches compute_observations:
            [contact_flag(8), contact_force(12), domain_rand(38), base_lin_vel(3),
             ang_vel(3), gravity(3), cmd(3), dof_pos(16), dof_vel(16),
             actions(16), heights(187)]
        """
        priv_dim = self.cfg.env.privileged_dim  # 61 (offset to actor/WM proprio)

        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level

        # base_lin_vel occupies [58:61] and is privileged-only.
        base_lin_offset = priv_dim - 3
        noise_vec[base_lin_offset:priv_dim] = (
            noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        )

        # Actor/WM proprio section starts at priv_dim (61).
        # Layout: [ang_vel(3), gravity(3), cmd(3), dof_pos(16), dof_vel(16), actions(16)]
        offset = priv_dim

        # ang_vel (3)
        noise_vec[offset:offset + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # gravity (3)
        noise_vec[offset + 3:offset + 6] = noise_scales.gravity * noise_level
        # commands (3): no noise
        # dof_pos (16)
        noise_vec[offset + 9:offset + 25] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # dof_vel (16)
        noise_vec[offset + 25:offset + 41] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # actions (16): no noise

        # Heights start after WM prop (41) and the separate action block (16).
        height_offset = priv_dim + self.cfg.env.prop_dim + self.cfg.env.action_dim
        noise_vec[height_offset:height_offset + self.cfg.env.height_dim] = (
            noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        )

        return noise_vec

    def get_amp_observations(self):
        """AMP observations for 16 DOF: joint_pos, base_lin_vel, base_ang_vel, joint_vel."""
        return torch.cat((
            self.dof_pos,
            self.base_lin_vel,
            self.base_ang_vel,
            self.dof_vel
        ), dim=-1)

    # ==================== Titatit-specific reward functions ====================

    def _reward_lateral_deviation(self):
        """Penalize lateral deviation from the origin corridor.
        Prevents the robot from going around obstacles instead of through them.
        """
        lateral_error = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
        deadband = self.lateral_deviation_deadband
        return torch.square(torch.clamp(lateral_error - deadband, min=0.0))

    def _reward_lateral_vel(self):
        """Penalize lateral (y-axis) velocity."""
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_tracking_lin_vel(self):
        """Track commanded velocity, but allow slower approach near obstacles."""
        target_vel = self.commands[:, :2].clone()
        if self.cfg.terrain.measure_heights:
            active = self._terrain_progress_active_mask()
            gap_active = self._gap_active_mask()
            obstacle_target_x = torch.where(
                gap_active,
                torch.full_like(target_vel[:, 0], self.gap_tracking_lin_vel_x),
                torch.full_like(target_vel[:, 0], self.obstacle_tracking_lin_vel_x),
            )
            target_vel[:, 0] = torch.where(
                active,
                torch.minimum(target_vel[:, 0], obstacle_target_x),
                target_vel[:, 0],
            )

        lin_vel = self.base_lin_vel[:, :2].clone()
        lin_vel_upper_bound = torch.where(target_vel < 0, 1e5, target_vel + self.cfg.rewards.lin_vel_clip)
        lin_vel_lower_bound = torch.where(target_vel > 0, -1e5, target_vel - self.cfg.rewards.lin_vel_clip)
        clip_lin_vel = torch.clip(lin_vel, lin_vel_lower_bound, lin_vel_upper_bound)
        lin_vel_error = torch.sum(torch.square(target_vel - clip_lin_vel), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_terrain_progress(self):
        """Reward forward progress only when an obstacle is visible ahead.

        A dense always-on progress reward lets the wheeled robot solve the
        curriculum by timing/rolling forward without relying on perception.
        Keep this signal tied to measured terrain variation so the policy is
        pushed to use the depth/world-model feature near gaps and steps.
        """
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        active = self._terrain_progress_active_mask()
        command_xy = self.commands[:, :2]
        command_norm = torch.norm(command_xy, dim=1, keepdim=True)
        command_dir = command_xy / torch.clamp(command_norm, min=1e-6)
        progress_vel = torch.sum(self.base_lin_vel[:, :2] * command_dir, dim=1)
        progress_vel = torch.clamp(progress_vel, min=0.0, max=self.terrain_progress_max_vel)
        moving_command = command_norm.squeeze(1) > 0.05

        lateral_error = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
        lateral_weight = torch.exp(-torch.square(lateral_error / self.terrain_corridor_deadband))

        forward = quat_apply_yaw(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        heading_weight = torch.exp(-torch.square(heading / self.terrain_corridor_heading_width))
        front_impact = self._front_wheel_impact_score() / self.terrain_progress_impact_width
        stair_safe_weight = torch.exp(-front_impact)
        pit_safe_weight = torch.exp(-self.pit_progress_impact_weight * front_impact)
        safe_weight = torch.where(self.pit_env_mask, pit_safe_weight, stair_safe_weight)

        return (
            active.float()
            * moving_command.float()
            * progress_vel
            * lateral_weight
            * heading_weight
            * safe_weight
        )

    def _reward_terrain_corridor(self):
        """Reward forward progress weighted by corridor alignment and heading.
        Combines forward velocity with lateral/heading penalties.
        """
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        active = self._terrain_progress_active_mask()
        command_forward = self.commands[:, 0] > 0.05
        progress_vel = torch.clamp(self.base_lin_vel[:, 0], min=0.0, max=self.terrain_progress_max_vel)

        # Lateral weight: exp(-lateral_error^2 / deadband^2)
        lateral_error = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
        lateral_weight = torch.exp(-torch.square(lateral_error / self.terrain_corridor_deadband))

        # Heading weight: exp(-heading^2 / width^2)
        forward = quat_apply_yaw(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        heading_weight = torch.exp(-torch.square(heading / self.terrain_corridor_heading_width))

        return active.float() * command_forward.float() * progress_vel * lateral_weight * heading_weight

    def _forward_height_patch(self, x0, x1, y_radius=None):
        forward_heights = self.measured_forward_heights.view(
            self.num_envs,
            len(self.cfg.terrain.measured_forward_points_x),
            len(self.cfg.terrain.measured_forward_points_y),
        )
        x0 = max(min(int(x0), forward_heights.shape[1] - 1), 0)
        x1 = max(min(int(x1), forward_heights.shape[1]), x0 + 1)
        if y_radius is None:
            y_radius = self.terrain_progress_y_radius
        center_y = forward_heights.shape[2] // 2
        y0 = max(center_y - y_radius, 0)
        y1 = min(center_y + y_radius + 1, forward_heights.shape[2])
        return forward_heights[:, x0:x1, y0:y1]

    def _terrain_variation_active_mask(self, x0, x1, y_radius=None, trigger_height=None):
        """Check if a forward height patch contains significant variation."""
        if trigger_height is None:
            trigger_height = self.terrain_progress_trigger_height

        front_patch = self._forward_height_patch(x0, x1, y_radius=y_radius)
        terrain_variation = (
            torch.max(front_patch, dim=2)[0].max(dim=1)[0]
            - torch.min(front_patch, dim=2)[0].min(dim=1)[0]
        )
        return terrain_variation > trigger_height

    def _terrain_progress_active_mask(self):
        """Check if there's significant terrain variation ahead (obstacle detected)."""
        return self._terrain_variation_active_mask(
            self.terrain_progress_x0,
            self.terrain_progress_x1,
        )

    def _early_front_clearance_active_mask(self):
        """Detect forward height discontinuities before the front wheels reach them."""
        return self._terrain_variation_active_mask(
            self.early_front_clearance_x0,
            self.early_front_clearance_x1,
            trigger_height=self.early_front_clearance_trigger_height,
        )

    def _front_obstacle_profile(self):
        front_patch = self._forward_height_patch(
            self.early_front_clearance_x0,
            self.early_front_clearance_x1,
        )
        x0 = max(min(int(self.early_front_clearance_x0), self.forward_height_x_values.numel() - 1), 0)
        x1 = max(min(int(self.early_front_clearance_x1), self.forward_height_x_values.numel()), x0 + 1)
        x_values = self.forward_height_x_values[x0:x1]

        support_ground = self._support_ground_height().unsqueeze(1)
        column_high = torch.max(front_patch, dim=2).values
        column_low = torch.min(front_patch, dim=2).values
        rise = torch.clamp(column_high - support_ground, min=0.0)
        drop = torch.clamp(support_ground - column_low, min=0.0)
        discontinuity = torch.maximum(rise, drop)

        obstacle_height = torch.max(discontinuity, dim=1).values
        obstacle_cols = discontinuity > self.early_front_clearance_trigger_height
        far_value = torch.full_like(discontinuity, 1e6)
        column_distance = torch.where(obstacle_cols, x_values.unsqueeze(0), far_value)
        obstacle_distance = torch.min(column_distance, dim=1).values
        obstacle_visible = obstacle_distance < 1e5
        obstacle_distance = torch.where(
            obstacle_visible,
            obstacle_distance,
            torch.full_like(obstacle_distance, self.obstacle_front_clearance_far_distance),
        )
        return obstacle_height, obstacle_distance, obstacle_visible

    def _front_obstacle_swing_phase(self):
        """Return a distance gate for useful obstacle-front leg swing."""
        obstacle_height, obstacle_distance, obstacle_visible = self._front_obstacle_profile()
        start = self.obstacle_front_swing_start_distance
        peak = self.obstacle_front_swing_peak_distance
        end = self.obstacle_front_swing_end_distance
        phase_rise = torch.clamp((start - obstacle_distance) / max(start - peak, 1e-3), 0.0, 1.0)
        phase_fall = torch.clamp((obstacle_distance - end) / max(peak - end, 1e-3), 0.0, 1.0)
        swing_phase = torch.minimum(phase_rise, phase_fall)
        swing_phase = swing_phase * (obstacle_distance < start).float()
        return obstacle_height, obstacle_distance, obstacle_visible, swing_phase

    def _front_height_change_profile(self):
        front_patch = self._forward_height_patch(
            self.early_front_clearance_x0,
            self.early_front_clearance_x1,
        )
        x0 = max(min(int(self.early_front_clearance_x0), self.forward_height_x_values.numel() - 1), 0)
        x1 = max(min(int(self.early_front_clearance_x1), self.forward_height_x_values.numel()), x0 + 1)
        x_values = self.forward_height_x_values[x0:x1]
        if x_values.numel() < 2:
            no_change = torch.zeros(self.num_envs, device=self.device)
            far_distance = torch.full_like(no_change, self.obstacle_front_clearance_far_distance)
            return no_change, far_distance, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        column_high = torch.max(front_patch, dim=2).values
        column_low = torch.min(front_patch, dim=2).values
        high_change = torch.abs(column_high[:, 1:] - column_high[:, :-1])
        low_change = torch.abs(column_low[:, 1:] - column_low[:, :-1])
        height_change = torch.maximum(high_change, low_change)

        max_change = torch.max(height_change, dim=1).values
        change_cols = height_change > self.obstacle_front_air_trigger_height
        far_value = torch.full_like(height_change, 1e6)
        change_distance = torch.where(change_cols, x_values[1:].unsqueeze(0), far_value)
        change_distance = torch.min(change_distance, dim=1).values
        change_visible = change_distance < 1e5
        change_distance = torch.where(
            change_visible,
            change_distance,
            torch.full_like(change_distance, self.obstacle_front_clearance_far_distance),
        )
        return max_change, change_distance, change_visible

    def _front_obstacle_clearance_deficit(self):
        obstacle_height, obstacle_distance, obstacle_visible = self._front_obstacle_profile()
        active = (
            self.high_obstacle_env_mask
            & obstacle_visible
            & (obstacle_height > self.early_front_clearance_trigger_height)
        )
        moving_forward = self.commands[:, 0] > 0.1

        support_front_clearance = self._foot_clearance_above_support(self.front_foot_indices)
        horizontal_force, vertical_force, _ = self._front_wheel_force_scores()
        front_light = vertical_force < self.front_clearance_light_contact_force
        no_face_contact = horizontal_force < self.front_clearance_horizontal_force
        useful_clearance = torch.max(
            support_front_clearance * front_light.float() * no_face_contact.float(),
            dim=1,
        ).values

        target = self._obstacle_front_clearance_target()
        raw_deficit = torch.clamp((target - useful_clearance) / target, min=0.0, max=1.0)
        distance_range = max(
            self.obstacle_front_clearance_far_distance - self.obstacle_front_clearance_deadline_distance,
            1e-3,
        )
        deadline_weight = torch.clamp(
            (self.obstacle_front_clearance_far_distance - obstacle_distance) / distance_range,
            min=0.0,
            max=1.0,
        )
        weighted_deficit = (0.25 + 0.75 * deadline_weight) * raw_deficit
        return active.float() * moving_forward.float() * weighted_deficit

    def _front_obstacle_edge_context(self):
        obstacle_height, obstacle_distance, obstacle_visible = self._front_obstacle_profile()
        active = (
            self.stairup_env_mask
            & obstacle_visible
            & (obstacle_height > self.early_front_clearance_trigger_height)
            & (self.commands[:, 0] > 0.1)
        )

        front_clearance = self._foot_clearance_above_support(self.front_foot_indices)
        clearance_target = self._obstacle_front_clearance_target().unsqueeze(1)
        lift_range = torch.clamp(clearance_target - self.obstacle_front_swing_min_lift, min=1e-3)
        clearance_score = torch.clamp(
            (front_clearance - self.obstacle_front_swing_min_lift) / lift_range,
            0.0,
            1.0,
        )
        best_clearance = torch.max(front_clearance, dim=1).values
        raw_deficit = torch.clamp(
            (clearance_target.squeeze(1) - best_clearance) / clearance_target.squeeze(1),
            0.0,
            1.0,
        )

        horizontal_force, vertical_force, _ = self._front_wheel_force_scores()
        front_contact = self._foot_contacts(self.front_foot_indices)
        front_air = torch.logical_not(front_contact)
        light_score = torch.clamp(
            (self.front_clearance_light_contact_force - vertical_force)
            / max(self.front_clearance_light_contact_force, 1e-3),
            0.0,
            1.0,
        )
        light_score = torch.maximum(light_score, front_air.float())
        clean_score = torch.clamp(
            (self.obstacle_front_edge_clean_force - horizontal_force)
            / max(self.obstacle_front_edge_clean_force, 1e-3),
            0.0,
            1.0,
        )
        max_horizontal_force = torch.max(horizontal_force, dim=1).values

        prepare_range = max(
            self.obstacle_front_edge_prepare_distance - self.obstacle_front_edge_deadline_distance,
            1e-3,
        )
        approach_weight = torch.clamp(
            (self.obstacle_front_edge_prepare_distance - obstacle_distance) / prepare_range,
            0.0,
            1.0,
        )
        close_range = max(
            self.obstacle_front_edge_deadline_distance - self.obstacle_front_edge_close_distance,
            1e-3,
        )
        close_weight = torch.clamp(
            (self.obstacle_front_edge_deadline_distance - obstacle_distance) / close_range,
            0.0,
            1.0,
        )
        return (
            active,
            approach_weight,
            close_weight,
            clearance_score,
            light_score,
            clean_score,
            raw_deficit,
            max_horizontal_force,
        )

    def _obstacle_active_mask(self, env_mask):
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return env_mask & self._terrain_progress_active_mask()

    def _stagger_obstacle_active_mask(self):
        return self._obstacle_active_mask(self.stagger_obstacle_env_mask)

    def _gap_active_mask(self):
        return self._obstacle_active_mask(self.gap_env_mask)

    def _step_obstacle_active_mask(self):
        return self._obstacle_active_mask(self.step_obstacle_env_mask)

    def _foot_contacts(self, foot_indices):
        return self.contact_forces[:, foot_indices, 2] > self.gait_contact_force_threshold

    def _foot_clearance(self, foot_indices):
        foot_pos = self.rigid_body_pos[:, foot_indices, :]
        if self.cfg.terrain.mesh_type == 'plane':
            return foot_pos[:, :, 2]

        points = foot_pos + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        ground_heights = torch.min(torch.min(heights1, heights2), heights3)
        ground_heights = ground_heights.view(self.num_envs, foot_indices.numel()) * self.terrain.cfg.vertical_scale
        return foot_pos[:, :, 2] - ground_heights

    def _support_ground_height(self):
        """Use nearby support height as reference, not gap/drop floor height."""
        return torch.max(self.measured_heights, dim=1).values

    def _foot_clearance_above_support(self, foot_indices):
        foot_pos = self.rigid_body_pos[:, foot_indices, :]
        support_ground = self._support_ground_height().unsqueeze(1)
        return foot_pos[:, :, 2] - support_ground

    def _front_obstacle_relative_heights(self):
        front_patch = self._forward_height_patch(
            self.early_front_clearance_x0,
            self.early_front_clearance_x1,
        )
        support_ground = self._support_ground_height()
        front_high = torch.max(front_patch, dim=2)[0].max(dim=1)[0]
        front_low = torch.min(front_patch, dim=2)[0].min(dim=1)[0]
        rise = torch.clamp(front_high - support_ground, min=0.0)
        drop = torch.clamp(support_ground - front_low, min=0.0)
        return rise, drop

    def _reward_orientation(self):
        """Penalize non-flat base orientation, with extra penalty for pitching up on slopes.

        In addition to the base class penalty on projected_gravity[:2],
        this adds extra penalty when the robot pitches up (nose up /抬头)
        on complex terrain (terrain_level > 2), which is a common issue on slopes.
        """
        # Base orientation penalty (same as base class)
        base_orient_penalty = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

        # Extra penalty for pitching up (抬头)
        # projected_gravity[:, 0] corresponds to pitch direction in base frame
        # Negative value means the base is pitching up (front rising)
        # Clamp to only penalize pitch-up (negative projected_gravity_x)
        pitch_up = torch.clamp(self.projected_gravity[:, 0], max=0.0)
        pitch_up_penalty = torch.square(pitch_up)

        # Apply extra pitch-up penalty on all terrain, but stronger on complex terrain
        # Base penalty always active; extra multiplier on terrain_level > 2 (slopes, stairs, etc.)
        slope_mask = 0.5 + 0.5 * (self.terrain_levels > 2).float()

        return base_orient_penalty + 5.0 * pitch_up_penalty * slope_mask

    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces -适用于所有地形类型.

        Overrides the base class implementation which only applies to gap/pit terrain.
        For the wheeled-legged robot, detects when feet experience high horizontal
        forces relative to vertical forces (stumbling/tripping).

        Unlike the base class, this applies to ALL terrain types including slopes,
        so the robot learns to avoid tripping on斜坡 terrain.
        """
        # Detect high horizontal force relative to vertical force at feet
        horizontal_force = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        vertical_force = torch.abs(self.contact_forces[:, self.feet_indices, 2])

        # A foot is stumbling if horizontal force > 2x vertical force AND vertical force > threshold
        # (ensuring the foot is actually in contact with something)
        stumble = (horizontal_force > 2.0 * vertical_force) & (vertical_force > 1.0)

        return torch.sum(stumble.float(), dim=1)

    def _front_wheel_force_scores(self):
        front_forces = self.contact_forces[:, self.front_foot_indices, :]
        horizontal_force = torch.norm(front_forces[:, :, :2], dim=2)
        vertical_force = torch.abs(front_forces[:, :, 2]) + 1e-6
        force_score = torch.clamp(
            (horizontal_force - self.front_wheel_impact_force_threshold)
            / self.front_wheel_impact_force_scale,
            min=0.0,
            max=1.0,
        )
        ratio_score = torch.clamp(
            (horizontal_force / vertical_force - self.front_wheel_impact_vertical_ratio)
            / self.front_wheel_impact_ratio_scale,
            min=0.0,
            max=1.0,
        )
        impact_score = torch.sqrt(force_score * ratio_score + 1e-8)
        impact_score = torch.where(
            (force_score > 0.0) & (ratio_score > 0.0),
            impact_score,
            torch.zeros_like(impact_score),
        )
        return horizontal_force, vertical_force, impact_score

    def _front_wheel_impact_score(self):
        _, _, impact_score = self._front_wheel_force_scores()
        speed_score = torch.clamp(
            (self.base_lin_vel[:, 0] - self.front_wheel_impact_speed_threshold)
            / self.front_wheel_impact_speed_scale,
            min=0.0,
            max=1.0,
        )
        impact_score = torch.sum(impact_score, dim=1) * speed_score
        return torch.clamp(impact_score, max=self.front_wheel_impact_max_penalty)

    def _controlled_front_contact_score(self):
        horizontal_force, vertical_force, impact_score = self._front_wheel_force_scores()
        front_contact = vertical_force > self.gait_contact_force_threshold
        force_ok = (
            (horizontal_force > self.controlled_front_contact_min_force)
            & (horizontal_force < self.controlled_front_contact_max_force)
        )
        ratio_ok = (horizontal_force / vertical_force) < self.controlled_front_contact_max_ratio
        low_impact = impact_score < 0.2
        contact_score = torch.max((front_contact & force_ok & ratio_ok & low_impact).float(), dim=1).values

        speed_error = (self.base_lin_vel[:, 0] - self.controlled_front_contact_speed_target)
        speed_weight = torch.exp(-torch.square(speed_error / self.controlled_front_contact_speed_width))
        rear_contact = self._foot_contacts(self.rear_foot_indices)
        rear_support = rear_contact[:, 0] & rear_contact[:, 1]
        return contact_score * speed_weight * rear_support.float()

    def _reward_front_wheel_impact(self):
        """Penalize hard front-wheel impacts without punishing normal rolling contact.

        Normal wheel-ground support has large vertical force, so this only fires
        when horizontal force is large, large relative to vertical support, and
        the robot is still moving quickly into the obstacle.
        """
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        active = self._terrain_progress_active_mask()
        moving_forward = self.commands[:, 0] > 0.1
        clearance_deficit = self._front_obstacle_clearance_deficit()
        pit_weight = torch.where(
            self.pit_env_mask,
            torch.full_like(clearance_deficit, self.pit_front_wheel_impact_weight),
            torch.ones_like(clearance_deficit),
        )
        return active.float() * moving_forward.float() * pit_weight * (1.0 + clearance_deficit) * self._front_wheel_impact_score()

    def _reward_front_touchdown_impact(self):
        """Penalize hard downward front-wheel touchdown on steps, drops, and gaps."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        active = self._step_obstacle_active_mask()
        moving_forward = self.commands[:, 0] > 0.1

        front_forces = self.contact_forces[:, self.front_foot_indices, :]
        vertical_force = torch.abs(front_forces[:, :, 2])
        front_contact = vertical_force > self.gait_contact_force_threshold
        force_score = torch.clamp(
            (vertical_force - self.front_touchdown_impact_force_threshold)
            / self.front_touchdown_impact_force_scale,
            min=0.0,
            max=1.0,
        )

        front_vel_z = self.rigid_body_lin_vel[:, self.front_foot_indices, 2]
        down_speed = torch.clamp(-front_vel_z, min=0.0)
        speed_score = torch.clamp(
            (down_speed - self.front_touchdown_impact_down_speed_threshold)
            / self.front_touchdown_impact_down_speed_scale,
            min=0.0,
            max=1.0,
        )
        touchdown_score = torch.max(front_contact.float() * force_score * speed_score, dim=1).values
        return active.float() * moving_forward.float() * touchdown_score

    def _reward_controlled_front_contact(self):
        """Small reward for low-speed, stable front-wheel contact at step edges."""
        active = self._stagger_obstacle_active_mask()
        moving_forward = self.commands[:, 0] > 0.1
        return active.float() * moving_forward.float() * self._controlled_front_contact_score()

    def _reward_obstacle_front_swing_phase(self):
        """Guide a clean front-wheel swing in the distance window before impact."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        obstacle_height, obstacle_distance, obstacle_visible = self._front_obstacle_profile()
        active = (
            self.high_obstacle_env_mask
            & obstacle_visible
            & (obstacle_height > self.early_front_clearance_trigger_height)
            & (self.commands[:, 0] > 0.1)
        )

        start = self.obstacle_front_swing_start_distance
        peak = self.obstacle_front_swing_peak_distance
        end = self.obstacle_front_swing_end_distance
        phase_rise = torch.clamp((start - obstacle_distance) / max(start - peak, 1e-3), 0.0, 1.0)
        phase_fall = torch.clamp((obstacle_distance - end) / max(peak - end, 1e-3), 0.0, 1.0)
        swing_phase = torch.minimum(phase_rise, phase_fall) * active.float()

        clearance = self._foot_clearance_above_support(self.front_foot_indices)
        clearance_target = torch.clamp(
            obstacle_height + self.obstacle_front_swing_clearance_margin,
            min=self.obstacle_front_swing_min_clearance,
            max=self.obstacle_front_swing_max_clearance,
        ).unsqueeze(1)
        lift_range = torch.clamp(clearance_target - self.obstacle_front_swing_min_lift, min=1e-3)
        clearance_score = torch.clamp(
            (clearance - self.obstacle_front_swing_min_lift) / lift_range,
            0.0,
            1.0,
        )

        horizontal_force, vertical_force, _ = self._front_wheel_force_scores()
        front_contact = self._foot_contacts(self.front_foot_indices)
        front_air = torch.logical_not(front_contact)
        unload_score = torch.clamp(
            (self.front_clearance_light_contact_force - vertical_force)
            / max(self.front_clearance_light_contact_force, 1e-3),
            0.0,
            1.0,
        )
        swing_candidate = torch.maximum(front_air.float(), unload_score)
        clean_swing = torch.clamp(
            (self.obstacle_front_swing_impact_threshold - horizontal_force)
            / max(self.obstacle_front_swing_impact_threshold, 1e-3),
            0.0,
            1.0,
        )

        rear_contact = self._foot_contacts(self.rear_foot_indices)
        rear_support = torch.sum(rear_contact.float(), dim=1)
        rear_support_weight = 0.5 + 0.25 * rear_support

        per_leg_score = swing_candidate * clean_swing * (0.25 + 0.75 * clearance_score)
        swing_score = torch.max(per_leg_score, dim=1).values
        return swing_phase * rear_support_weight * swing_score

    def _reward_obstacle_front_edge_clearance(self):
        """Reward front wheels that clear the upcoming edge before face contact."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        (
            active,
            approach_weight,
            _,
            clearance_score,
            light_score,
            clean_score,
            _,
            max_horizontal_force,
        ) = self._front_obstacle_edge_context()

        per_leg_score = clearance_score * light_score * clean_score
        max_leg_score = torch.max(per_leg_score, dim=1).values
        mean_leg_score = torch.mean(per_leg_score, dim=1)
        global_clean = torch.clamp(
            (self.obstacle_front_edge_clean_force - max_horizontal_force)
            / max(self.obstacle_front_edge_clean_force, 1e-3),
            0.0,
            1.0,
        )
        edge_clearance_score = (0.7 * max_leg_score + 0.3 * mean_leg_score) * global_clean
        progress_score = 0.15 + 0.85 * torch.clamp(
            self.base_lin_vel[:, 0] / max(self.obstacle_front_edge_progress_vel, 1e-3),
            0.0,
            1.0,
        )
        stability_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        stability_weight = torch.exp(-stability_error / self.front_swing_stability_width)
        return active.float() * approach_weight * edge_clearance_score * progress_score * stability_weight

    def _reward_obstacle_front_edge_fail(self):
        """Penalize arriving at the edge without clearance, or stopping there."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        (
            active,
            _,
            close_weight,
            _,
            _,
            _,
            raw_deficit,
            max_horizontal_force,
        ) = self._front_obstacle_edge_context()

        force_fail = torch.clamp(
            (max_horizontal_force - self.obstacle_front_edge_clean_force)
            / max(self.obstacle_front_edge_clean_force, 1e-3),
            0.0,
            1.0,
        )
        impact_fail = torch.clamp(
            self._front_wheel_impact_score() / max(self.front_wheel_impact_max_penalty, 1e-3),
            0.0,
            1.0,
        )
        stuck_fail = torch.clamp(
            (self.obstacle_front_edge_stuck_speed - self.base_lin_vel[:, 0])
            / max(self.obstacle_front_edge_stuck_speed, 1e-3),
            0.0,
            1.0,
        )
        fail_score = torch.clamp(
            0.70 * raw_deficit
            + 0.45 * force_fail
            + 0.45 * impact_fail
            + 0.35 * stuck_fail * raw_deficit,
            0.0,
            1.5,
        )
        return active.float() * close_weight * fail_score

    def _reward_feet_edge(self):
        """Penalize wheel/foot contacts on sharp gap and pit edges.

        The base implementation depends on ``contact_filt`` created by
        ``_reward_feet_air_time``. Keep this term self-contained so it remains
        stable when the feet-air-time scale is changed.
        """
        feet_pos_xy = (
            (self.rigid_body_states.view(self.num_envs, -1, 13)[:, self.feet_indices, :2]
             + self.terrain.cfg.border_size)
            / self.cfg.terrain.horizontal_scale
        ).round().long()
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.feet_at_edge = contact_filt & feet_at_edge

        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        edge_reward = torch.zeros_like(rew)
        edge_reward[self.gap_start_idx:self.pit_end_idx] = rew[self.gap_start_idx:self.pit_end_idx]
        return edge_reward

    def _reward_front_pair_air(self):
        """Penalize hopping with both front wheels lifted at the same time.

        On stairs and climb-style obstacles the policy was succeeding by
        pitching up and moving both front legs together. This term keeps the
        front legs staggered there, while gap terrain is handled separately.
        """
        active = self._stagger_obstacle_active_mask()
        moving_forward = self.commands[:, 0] > 0.1

        front_contact = self._foot_contacts(self.front_foot_indices)
        both_front_air = torch.logical_not(front_contact[:, 0]) & torch.logical_not(front_contact[:, 1])

        front_clearance = self._foot_clearance(self.front_foot_indices)
        both_front_high = torch.min(front_clearance, dim=1).values > self.front_stagger_clearance_threshold

        return active.float() * moving_forward.float() * (
            both_front_air.float() + both_front_high.float()
        )

    def _reward_obstacle_front_lift(self):
        """Reward anticipatory front clearance on stair/pit without prescribing gait.

        This rewards useful front-wheel height before impact, whether the policy
        lifts one front wheel first or both front wheels together.
        """
        obstacle_height, _, obstacle_visible, swing_phase = self._front_obstacle_swing_phase()
        active = (
            self.stagger_obstacle_env_mask
            & obstacle_visible
            & (obstacle_height > self.early_front_clearance_trigger_height)
            & (swing_phase > 0.0)
        )
        moving_forward = self.commands[:, 0] > 0.1

        # 获取前方实际障碍物高度，并加上设定的裕度
        target_clearance = torch.clamp(
            obstacle_height + self.obstacle_front_swing_clearance_margin,
            min=self.obstacle_front_swing_min_clearance,
            max=self.obstacle_front_swing_max_clearance
        )

        front_clearance = self._foot_clearance(self.front_foot_indices)
        useful_clearance = torch.max(front_clearance, dim=1).values
        # 到达目标高度拿满分，但如果过度高抬 (超过5cm) 则惩罚衰减，避免不必要的“砸地”
        overshoot = torch.clamp(useful_clearance - target_clearance, min=0.0)
        clearance_score = torch.clamp(useful_clearance / target_clearance, 0.0, 1.0) * torch.exp(-torch.square(overshoot / 0.05))

        front_forces = self.contact_forces[:, self.front_foot_indices, :]
        front_horizontal_force = torch.norm(front_forces[:, :, :2], dim=2)
        low_impact = torch.max(front_horizontal_force, dim=1).values < self.early_front_clearance_impact_threshold

        stability_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        stability_weight = torch.exp(-stability_error / self.front_swing_stability_width)

        return (
            active.float()
            * moving_forward.float()
            * swing_phase
            * low_impact.float()
            * clearance_score
            * stability_weight
        )

    def _reward_front_swing_clearance(self):
        """Reward a single front wheel lifting enough to clear an obstacle.

        This gives the robot a legged alternative to simply driving into the
        step: lift one front wheel, place it, then let the other side follow.
        """
        obstacle_height, _, obstacle_visible, swing_phase = self._front_obstacle_swing_phase()
        active = (
            self._stagger_obstacle_active_mask()
            & obstacle_visible
            & (obstacle_height > self.early_front_clearance_trigger_height)
            & (swing_phase > 0.0)
        )
        moving_forward = self.commands[:, 0] > 0.1

        front_contact = self._foot_contacts(self.front_foot_indices)
        front_air = torch.logical_not(front_contact)
        single_front_swing = front_air[:, 0] ^ front_air[:, 1]
        rear_contact = self._foot_contacts(self.rear_foot_indices)
        rear_support = rear_contact[:, 0] & rear_contact[:, 1]

        front_clearance = self._foot_clearance(self.front_foot_indices)
        swing_clearance = torch.max(front_clearance * front_air.float(), dim=1).values
        clearance_score = torch.clamp(swing_clearance / self.front_swing_clearance_target, 0.0, 1.0)
        stability_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        stability_weight = torch.exp(-stability_error / self.front_swing_stability_width)

        return (
            active.float()
            * moving_forward.float()
            * swing_phase
            * single_front_swing.float()
            * rear_support.float()
            * clearance_score
            * stability_weight
        )

    def _reward_no_obstacle_front_air(self):
        """Penalize unnecessary front-wheel lift when no obstacle is visible."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        no_obstacle = torch.logical_not(self._terrain_progress_active_mask())
        moving_forward = self.commands[:, 0] > self.no_obstacle_front_air_speed_threshold

        front_contact_force = self.contact_forces[:, self.front_foot_indices, 2]
        front_air = front_contact_force < self.no_obstacle_front_air_contact_force
        front_clearance = self._foot_clearance_above_support(self.front_foot_indices)
        excess_clearance = torch.clamp(
            front_clearance - self.no_obstacle_front_air_clearance,
            min=0.0,
        )
        height_penalty = torch.clamp(
            excess_clearance / max(self.no_obstacle_front_air_height_width, 1e-3),
            0.0,
            1.0,
        )
        front_lift_penalty = torch.sum(
            torch.maximum(front_air.float(), height_penalty),
            dim=1,
        )

        return no_obstacle.float() * moving_forward.float() * front_lift_penalty

    def _reward_obstacle_dead_leg(self):
        """Penalize long, high, non-productive leg extension near obstacles."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        obstacle_height, _, obstacle_visible = self._front_obstacle_profile()
        active = (
            self.front_obstacle_air_env_mask
            & obstacle_visible
            & (obstacle_height > self.early_front_clearance_trigger_height)
            & (self.commands[:, 0] > 0.1)
        )
        active_legs = active.unsqueeze(1)

        contact = self.contact_forces[:, self.obstacle_foot_indices, 2] > self.gait_contact_force_threshold
        air = torch.logical_not(contact)
        air_time = torch.where(
            active_legs & air,
            self.obstacle_foot_air_time + self.dt,
            torch.zeros_like(self.obstacle_foot_air_time),
        )
        self.obstacle_foot_air_time = air_time

        front_limit = torch.full(
            (self.num_envs, 2),
            self.obstacle_dead_leg_front_air_time,
            device=self.device,
        )
        front_limit = torch.where(
            self.gap_env_mask.unsqueeze(1),
            torch.full_like(front_limit, self.obstacle_front_air_gap_max_time),
            front_limit,
        )
        rear_limit = torch.full(
            (self.num_envs, 2),
            self.obstacle_dead_leg_rear_air_time,
            device=self.device,
        )
        air_limit = torch.cat((front_limit, rear_limit), dim=1)
        long_air = torch.clamp(
            (air_time - air_limit) / torch.clamp(air_limit, min=1e-3),
            0.0,
            1.0,
        )

        clearance = self._foot_clearance_above_support(self.obstacle_foot_indices)
        high_air = torch.clamp(
            (clearance - self.obstacle_dead_leg_clearance)
            / max(self.obstacle_dead_leg_clearance_width, 1e-3),
            0.0,
            1.0,
        )

        leg_error = torch.sum(
            torch.square(
                self.dof_pos[:, self.obstacle_leg_joint_indices]
                - self.default_dof_pos[:, self.obstacle_leg_joint_indices]
            ),
            dim=2,
        )
        extension_score = torch.clamp(
            (leg_error - self.obstacle_dead_leg_dof_error)
            / max(self.obstacle_dead_leg_dof_width, 1e-3),
            0.0,
            1.0,
        )

        progress_gate = torch.clamp(
            (self.obstacle_dead_leg_min_progress_vel - self.base_lin_vel[:, 0])
            / max(self.obstacle_dead_leg_min_progress_vel, 1e-3),
            0.0,
            1.0,
        ).unsqueeze(1)
        penalty = air.float() * long_air * torch.maximum(high_air, extension_score)
        penalty = penalty * (0.5 + 0.5 * progress_gate)
        return active.float() * torch.sum(penalty, dim=1)

    def _reward_feet_air_time(self):
        """Reward useful front-wheel swing timing before visible terrain changes.

        This is separate from the base feet-air-time reward: it only accumulates
        while a forward height discontinuity is visible, uses front wheels only,
        and rejects lift that starts after the front wheel has already hit the
        obstacle face.
        """
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        obstacle_height, obstacle_distance, obstacle_visible = self._front_obstacle_profile()
        height_change, change_distance, change_visible = self._front_height_change_profile()
        obstacle_distance = torch.where(change_visible, change_distance, obstacle_distance)
        active = (
            obstacle_visible
            & change_visible
            & (obstacle_height > self.obstacle_front_air_trigger_height)
            & (height_change > self.obstacle_front_air_trigger_height)
            & (self.commands[:, 0] > 0.1)
        )
        active_legs = active.unsqueeze(1)

        front_contact = self._foot_contacts(self.front_foot_indices)
        contact_filt = torch.logical_or(front_contact, self.last_front_obstacle_contacts)
        front_air = torch.logical_not(front_contact)
        front_forces = self.contact_forces[:, self.front_foot_indices, :]
        horizontal_force = torch.norm(front_forces[:, :, :2], dim=2)
        vertical_force = torch.abs(front_forces[:, :, 2])
        _, _, impact_score = self._front_wheel_force_scores()
        face_contact_leg = active_legs & (
            (horizontal_force > self.obstacle_front_air_preimpact_force)
            | (impact_score > self.obstacle_front_air_preimpact_score)
        )
        face_contact = torch.any(face_contact_leg, dim=1, keepdim=True).expand_as(face_contact_leg)
        blocked = self.front_obstacle_air_blocked | face_contact
        preimpact_active_legs = active_legs & torch.logical_not(blocked)

        clearance = self._foot_clearance_above_support(self.front_foot_indices)
        sequence_active = self.front_obstacle_air_active | (preimpact_active_legs & front_air)
        air_tracking = sequence_active & front_air
        air_time = torch.where(
            air_tracking,
            self.front_obstacle_air_time + self.dt,
            self.front_obstacle_air_time,
        )
        air_max_clearance = torch.where(
            air_tracking,
            torch.maximum(self.front_obstacle_air_max_clearance, clearance),
            self.front_obstacle_air_max_clearance,
        )

        first_contact = (
            sequence_active
            & contact_filt
            & torch.logical_not(blocked)
            & (air_time > self.obstacle_front_air_min_time)
        )

        target_time = torch.full_like(obstacle_distance, self.obstacle_front_air_stair_target_time)
        target_time = torch.where(
            self.gap_env_mask,
            torch.full_like(target_time, self.obstacle_front_air_gap_target_time),
            target_time,
        )
        target_time = torch.where(
            self.pit_env_mask,
            torch.full_like(target_time, self.obstacle_front_air_pit_target_time),
            target_time,
        ).unsqueeze(1)
        max_time = torch.full_like(obstacle_distance, self.obstacle_front_air_max_time)
        max_time = torch.where(
            self.gap_env_mask,
            torch.full_like(max_time, self.obstacle_front_air_gap_max_time),
            max_time,
        ).unsqueeze(1)

        rise_width = torch.clamp(target_time - self.obstacle_front_air_min_time, min=1e-3)
        fall_width = torch.clamp(max_time - target_time, min=1e-3)
        early_score = torch.clamp(
            (air_time - self.obstacle_front_air_min_time) / rise_width,
            0.0,
            1.0,
        )
        late_score = torch.clamp((max_time - air_time) / fall_width, 0.0, 1.0)
        air_score = early_score * late_score

        obstacle_clearance_target = self._obstacle_front_clearance_target()
        gap_clearance_target = self._gap_front_clearance_target()
        clearance_target = torch.where(
            self.gap_env_mask,
            gap_clearance_target,
            obstacle_clearance_target,
        ).unsqueeze(1)
        lift_range = torch.clamp(clearance_target - self.obstacle_front_swing_min_lift, min=1e-3)
        clearance_score = torch.clamp(
            (air_max_clearance - self.obstacle_front_swing_min_lift) / lift_range,
            0.0,
            1.0,
        )
        current_clearance_score = torch.clamp(
            (clearance - self.obstacle_front_swing_min_lift) / lift_range,
            0.0,
            1.0,
        )

        unload_score = torch.clamp(
            (self.front_clearance_light_contact_force - vertical_force)
            / max(self.front_clearance_light_contact_force, 1e-3),
            0.0,
            1.0,
        )
        swing_candidate = torch.maximum(front_air.float(), unload_score)
        clean_swing = torch.clamp(
            (self.obstacle_front_swing_impact_threshold - horizontal_force)
            / max(self.obstacle_front_swing_impact_threshold, 1e-3),
            0.0,
            1.0,
        )
        clean_landing = torch.clamp(
            (self.obstacle_front_air_landing_force - horizontal_force)
            / max(self.obstacle_front_air_landing_force, 1e-3),
            0.0,
            1.0,
        )
        front_vel = self.rigid_body_lin_vel[:, self.front_foot_indices, :]
        lateral_speed = torch.abs(front_vel[:, :, 1])
        down_speed = torch.clamp(-front_vel[:, :, 2], min=0.0)
        lateral_score = torch.clamp(
            (self.obstacle_front_air_lateral_speed - lateral_speed)
            / max(self.obstacle_front_air_lateral_speed, 1e-3),
            0.0,
            1.0,
        )
        down_score = torch.clamp(
            (self.obstacle_front_air_down_speed - down_speed)
            / max(self.obstacle_front_air_down_speed, 1e-3),
            0.0,
            1.0,
        )
        clean_landing = clean_landing * lateral_score * down_score

        per_leg_score = first_contact.float() * air_score * clearance_score * clean_landing
        event_score = torch.clamp(torch.sum(per_leg_score, dim=1), 0.0, 1.0)

        start = self.obstacle_front_swing_start_distance
        peak = self.obstacle_front_swing_peak_distance
        end = self.obstacle_front_swing_end_distance
        phase_rise = torch.clamp((start - obstacle_distance) / max(start - peak, 1e-3), 0.0, 1.0)
        phase_fall = torch.clamp((obstacle_distance - end) / max(peak - end, 1e-3), 0.0, 1.0)
        swing_phase = torch.minimum(phase_rise, phase_fall)
        per_leg_prepare = (
            preimpact_active_legs.float()
            * swing_phase.unsqueeze(1)
            * swing_candidate
            * clean_swing
            * (0.25 + 0.75 * current_clearance_score)
        )
        prepare_score = torch.max(per_leg_prepare, dim=1).values

        progress_score = torch.clamp(
            self.base_lin_vel[:, 0] / max(self.obstacle_front_air_progress_vel, 1e-3),
            0.0,
            1.0,
        )
        progress_score = 0.25 + 0.75 * progress_score
        stability_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        stability_weight = torch.exp(-stability_error / self.front_swing_stability_width)
        terrain_weight = torch.full_like(event_score, self.obstacle_front_air_stair_weight)
        terrain_weight = torch.where(
            self.gap_env_mask,
            torch.full_like(terrain_weight, self.obstacle_front_air_gap_weight),
            terrain_weight,
        )
        terrain_weight = torch.where(
            self.pit_env_mask,
            torch.full_like(terrain_weight, self.obstacle_front_air_pit_weight),
            terrain_weight,
        )

        reset_air = contact_filt | (torch.logical_not(active_legs) & torch.logical_not(sequence_active))
        self.front_obstacle_air_time = torch.where(reset_air, torch.zeros_like(air_time), air_time)
        self.front_obstacle_air_max_clearance = torch.where(
            reset_air,
            torch.zeros_like(air_max_clearance),
            air_max_clearance,
        )
        self.front_obstacle_air_active = torch.where(
            reset_air,
            torch.zeros_like(sequence_active),
            sequence_active,
        )
        self.front_obstacle_air_blocked = torch.where(
            torch.logical_not(active_legs),
            torch.zeros_like(blocked),
            blocked,
        )
        self.last_front_obstacle_contacts = front_contact

        # === v23: 地形感知的前腿策略 ===
        # stair_up: 奖励单前腿先抬，抑制双前腿同时高抬
        # pit: 不强制单腿，允许双腿蹦，但需要后腿支撑
        # gap: 允许双前腿一起抬
        front_contact_local = self._foot_contacts(self.front_foot_indices)
        single_front_air = front_contact_local[:, 0] ^ front_contact_local[:, 1]  # XOR: 只有一条前腿离地

        front_clearance_local = self._foot_clearance_above_support(self.front_foot_indices)
        both_front_high = torch.min(front_clearance_local, dim=1).values > self.front_stagger_clearance_threshold

        # stair_up: 单腿优先
        single_front_score = single_front_air.float()
        prepare_score = torch.where(
            self.stairup_env_mask,
            prepare_score * (0.4 + 0.6 * single_front_score),
            prepare_score,
        )
        # stair_up: 双前腿都高抬则惩罚
        prepare_score = torch.where(
            self.stairup_env_mask & both_front_high,
            prepare_score * 0.3,
            prepare_score,
        )

        # 后腿支撑条件（所有地形通用）
        rear_contact_local = self._foot_contacts(self.rear_foot_indices)
        rear_support = (rear_contact_local[:, 0] | rear_contact_local[:, 1]).float()
        rear_support_weight = 0.5 + 0.5 * rear_support
        prepare_score = prepare_score * rear_support_weight

        return (0.55 * prepare_score + 0.45 * event_score) * progress_score * stability_weight * terrain_weight

    def _reward_early_front_swing_clearance(self):
        """Reward single-front-wheel lift before the obstacle is close.

        This uses a farther lookahead window than ``front_swing_clearance`` and
        suppresses reward when the front wheels are already taking horizontal
        impact, so contact-driven lifting is not treated as early anticipation.
        """
        active = self.stagger_obstacle_env_mask & self._early_front_clearance_active_mask()
        moving_forward = self.commands[:, 0] > 0.1

        front_contact = self._foot_contacts(self.front_foot_indices)
        front_air = torch.logical_not(front_contact)
        single_front_swing = front_air[:, 0] ^ front_air[:, 1]
        rear_contact = self._foot_contacts(self.rear_foot_indices)
        rear_support = rear_contact[:, 0] & rear_contact[:, 1]

        front_clearance = self._foot_clearance(self.front_foot_indices)
        swing_clearance = torch.max(front_clearance * front_air.float(), dim=1).values
        clearance_score = torch.clamp(swing_clearance / self.front_swing_clearance_target, 0.0, 1.0)

        front_forces = self.contact_forces[:, self.front_foot_indices, :]
        front_horizontal_force = torch.norm(front_forces[:, :, :2], dim=2)
        low_impact = torch.max(front_horizontal_force, dim=1).values < self.early_front_clearance_impact_threshold

        stability_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        stability_weight = torch.exp(-stability_error / self.front_swing_stability_width)

        return (
            active.float()
            * moving_forward.float()
            * single_front_swing.float()
            * rear_support.float()
            * low_impact.float()
            * clearance_score
            * stability_weight
        )

    def _reward_gap_front_pair_clearance(self):
        """Reward paired front-wheel clearance only on gap terrain.

        Gap crossing is different from stair climbing: a short paired front-leg
        hop can be useful, but it should still be tied to forward progress and
        centered heading instead of just rewarding airtime.
        """
        active = self._gap_active_mask()
        moving_forward = self.commands[:, 0] > 0.1

        front_clearance = self._foot_clearance_above_support(self.front_foot_indices)
        paired_clearance = torch.min(front_clearance, dim=1).values
        clearance_score = torch.clamp(paired_clearance / self._gap_front_clearance_target(), 0.0, 1.0)

        progress_score = torch.clamp(self.base_lin_vel[:, 0] / self.terrain_progress_max_vel, 0.0, 1.0)

        lateral_error = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
        lateral_weight = torch.exp(-torch.square(lateral_error / self.terrain_corridor_deadband))
        forward = quat_apply_yaw(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        heading_weight = torch.exp(-torch.square(heading / self.terrain_corridor_heading_width))

        return (
            active.float()
            * moving_forward.float()
            * clearance_score
            * progress_score
            * lateral_weight
            * heading_weight
        )

    def _gap_front_clearance_target(self):
        front_patch = self._forward_height_patch(
            self.early_front_clearance_x0,
            self.early_front_clearance_x1,
        )
        gap_floor = torch.min(front_patch, dim=2)[0].min(dim=1)[0]
        gap_drop = torch.clamp(self._support_ground_height() - gap_floor, min=0.0)
        return torch.clamp(
            gap_drop + self.gap_front_clearance_margin,
            min=self.gap_front_clearance_min_target,
            max=self.gap_front_clearance_max_target,
        )

    def _obstacle_front_clearance_target(self):
        rise, drop = self._front_obstacle_relative_heights()
        obstacle_height = torch.maximum(rise, drop)
        return torch.clamp(
            obstacle_height + self.obstacle_front_clearance_margin,
            min=self.obstacle_front_clearance_min_target,
            max=self.obstacle_front_clearance_max_target,
        )

    def _reward_obstacle_front_clearance(self):
        """Penalize seeing a rising obstacle while the front wheels remain low and loaded."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        deficit = self._front_obstacle_clearance_deficit()
        return torch.square(deficit)

    def _reward_stair_front_clearance_deadline(self):
        """Penalize reaching a stair edge without useful front-wheel clearance."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        obstacle_height, obstacle_distance, obstacle_visible = self._front_obstacle_profile()
        active = (
            self.stairup_env_mask
            & obstacle_visible
            & (obstacle_height > self.early_front_clearance_trigger_height)
            & (self.commands[:, 0] > 0.1)
        )

        front_clearance = self._foot_clearance_above_support(self.front_foot_indices)
        best_clearance = torch.max(front_clearance, dim=1).values
        target = self._obstacle_front_clearance_target()
        clearance_deficit = torch.clamp((target - best_clearance) / torch.clamp(target, min=1e-3), 0.0, 1.0)

        deadline_range = max(
            self.stair_front_clearance_deadline_distance - self.stair_front_clearance_close_distance,
            1e-3,
        )
        close_weight = torch.clamp(
            (self.stair_front_clearance_deadline_distance - obstacle_distance) / deadline_range,
            0.0,
            1.0,
        )
        return active.float() * close_weight * torch.square(clearance_deficit)

    def _reward_obstacle_base_clearance(self):
        """Penalize keeping the base too low while climbing obstacles."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        active = self._step_obstacle_active_mask()
        moving_forward = self.commands[:, 0] > 0.1
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        clearance_deficit = torch.clamp(
            (self.obstacle_base_height_target - base_height) / self.obstacle_base_height_margin,
            min=0.0,
            max=1.0,
        )
        return active.float() * moving_forward.float() * clearance_deficit

    def _reward_obstacle_posture(self):
        """Penalize ugly roll/pitch excursions only while approaching obstacles."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        obstacle_height, _, obstacle_visible = self._front_obstacle_profile()
        active = (
            self.front_obstacle_air_env_mask
            & obstacle_visible
            & (obstacle_height > self.early_front_clearance_trigger_height)
            & (self.commands[:, 0] > 0.1)
        )
        posture_error = torch.norm(self.projected_gravity[:, :2], dim=1)
        excess = torch.clamp(
            (posture_error - self.obstacle_posture_limit)
            / max(self.obstacle_posture_width, 1e-3),
            0.0,
            1.0,
        )
        return active.float() * torch.square(excess)

    def _reward_obstacle_ang_vel_xy(self):
        """Penalize sharp roll/pitch motion only near visible obstacles."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        obstacle_height, _, obstacle_visible = self._front_obstacle_profile()
        active = (
            self.front_obstacle_air_env_mask
            & obstacle_visible
            & (obstacle_height > self.early_front_clearance_trigger_height)
            & (self.commands[:, 0] > 0.1)
        )
        ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        excess = torch.clamp(
            (ang_vel_xy - self.obstacle_ang_vel_xy_limit)
            / max(self.obstacle_ang_vel_xy_width, 1e-3),
            0.0,
            1.0,
        )
        return active.float() * torch.square(excess)

    def _reward_base_height_low(self):
        """Penalize crouching too low without constraining upward clearance."""
        if self.cfg.terrain.measure_heights:
            base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        else:
            base_height = self.root_states[:, 2]
        deficit = torch.clamp(
            (self.base_height_low_target - base_height) / self.base_height_low_margin,
            min=0.0,
            max=1.0,
        )
        return torch.square(deficit)

    def _reward_cheat(self):
        """Penalize going around obstacles (heading deviation > threshold)."""
        forward = quat_apply_yaw(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        cheat = (torch.abs(heading) > 1.0).float()
        # Only apply when there's terrain variation ahead
        if self.cfg.terrain.measure_heights:
            active = self._terrain_progress_active_mask()
            return active.float() * cheat
        return cheat

    def _reward_stuck(self):
        """Penalize no progress under a forward command."""
        slow = torch.abs(self.base_lin_vel[:, 0]) < 0.08
        moving_command = torch.abs(self.commands[:, 0]) > 0.1
        pit_weight = torch.where(
            self.pit_env_mask,
            torch.full((self.num_envs,), self.pit_stuck_multiplier, device=self.device),
            torch.ones(self.num_envs, device=self.device),
        )
        return (slow & moving_command).float() * pit_weight

    def _reward_wheel_vel(self):
        """Reward wheel rotation when moving forward.
        Encourages the robot to use wheels instead of just walking.
        Wheel joints in Isaac order: indices 3,7,11,15.
        """
        wheel_vel = self.dof_vel[:, self.wheel_indices]
        avg_wheel_speed = torch.mean(torch.abs(wheel_vel), dim=1)
        # Only reward useful wheel rotation. Spinning wheels while stuck should not pay.
        moving_forward = self.commands[:, 0] > 0.1
        forward_speed = torch.clamp(self.base_lin_vel[:, 0], min=0.0)
        command_speed = torch.clamp(self.commands[:, 0], min=0.2)
        speed_gate = torch.clamp(forward_speed / command_speed, min=0.0, max=1.0)
        return moving_forward.float() * speed_gate * torch.clamp(avg_wheel_speed, max=10.0) / 10.0

    def _reward_wheel_idle(self):
        """Penalize idle wheels when moving forward.
        If the robot has a forward command but wheels are nearly stationary, penalize.
        """
        wheel_vel = torch.abs(self.dof_vel[:, self.wheel_indices])
        avg_wheel_speed = torch.mean(wheel_vel, dim=1)
        moving_forward = self.commands[:, 0] > 0.1
        idle = (avg_wheel_speed < 0.5).float()
        return moving_forward.float() * idle

    def _reward_dof_error(self):
        """Penalize leg joint deviation from defaults; wheel angles rotate continuously."""
        return torch.sum(
            torch.square(self.dof_pos[:, self.leg_indices] - self.default_dof_pos[:, self.leg_indices]),
            dim=1,
        )

    def _reward_hip_deviation(self):
        """Penalize long-horizon hip splay without constraining knee/calf stair motion."""
        return torch.sum(
            torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]),
            dim=1,
        )

    def _reward_rear_leg_deviation(self):
        """Keep rear support legs near their nominal shape without restricting front stepping."""
        return torch.sum(
            torch.square(self.dof_pos[:, self.rear_leg_indices] - self.default_dof_pos[:, self.rear_leg_indices]),
            dim=1,
        )
