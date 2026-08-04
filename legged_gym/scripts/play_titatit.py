# import os
# import inspect
# import time

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, parentdir)
# from legged_gym import LEGGED_GYM_ROOT_DIR

# import isaacgym
# from isaacgym import gymapi
# from legged_gym.envs import *
# from legged_gym.utils import get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

# try:
#     import cv2
# except ImportError:
#     cv2 = None

# import numpy as np
# import torch


# def fix_terrain_level(env, terrain_level):
#     if terrain_level is None:
#         return
#     if not hasattr(env, "terrain_levels"):
#         print("--terrain_level ignored: environment has no terrain_levels")
#         return

#     level = max(0, min(int(terrain_level), int(env.max_terrain_level) - 1))
#     env.terrain_levels[:] = level
#     env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
#     print(f"Fixed play terrain level to {level} for all envs")


# def create_recording_camera(env, env_id, width=512, height=512):
#     camera_local_transform = gymapi.Transform()
#     camera_local_transform.p = gymapi.Vec3(-0.5, -1.0, 0.1)
#     camera_local_transform.r = gymapi.Quat.from_axis_angle(
#         gymapi.Vec3(0, 0, 1),
#         np.deg2rad(90),
#     )

#     camera_props = gymapi.CameraProperties()
#     camera_props.width = width
#     camera_props.height = height

#     cam_handle = env.gym.create_camera_sensor(env.envs[env_id], camera_props)
#     body_handle = env.gym.get_actor_rigid_body_handle(env.envs[env_id], env.actor_handles[env_id], 0)
#     env.gym.attach_camera_to_body(
#         cam_handle,
#         env.envs[env_id],
#         body_handle,
#         camera_local_transform,
#         gymapi.FOLLOW_TRANSFORM,
#     )
#     return cam_handle


# def play(args):
#     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
#     # override for testing
#     env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else 1
#     env_cfg.terrain.num_cols = 1
#     env_cfg.terrain.curriculum = False
#     env_cfg.noise.add_noise = False

#     env_cfg.domain_rand.friction_range = [1.0, 1.0]
#     env_cfg.domain_rand.restitution_range = [0.0, 0.0]
#     env_cfg.domain_rand.added_mass_range = [0., 0.]
#     env_cfg.domain_rand.com_x_pos_range = [-0.0, 0.0]
#     env_cfg.domain_rand.com_y_pos_range = [-0.0, 0.0]
#     env_cfg.domain_rand.com_z_pos_range = [-0.0, 0.0]
#     env_cfg.domain_rand.randomize_action_latency = False
#     env_cfg.domain_rand.push_robots = False
#     env_cfg.domain_rand.randomize_gains = True
#     env_cfg.domain_rand.randomize_link_mass = False
#     env_cfg.domain_rand.randomize_motor_strength = False
#     env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
#     env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]

#     train_cfg.runner.amp_num_preload_transitions = 1
#     env_cfg.asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/titatit/urdf/wheeled_titatit_play.urdf'
#     env_cfg.asset.replace_cylinder_with_capsule = False

#     # terrain: 10 types for titatit
#     # [wave, rough_slope, stairs_up, stairs_down, discrete, gap, pit, tilt, crawl, rough_flat]
#     terrain_aliases = {
#         'stair_up': 'stair',
#         'stairs_up': 'stair',
#         'rough_flat': 'rough',
#         'climb': 'pit',
#     }
#     args.terrain = terrain_aliases.get(args.terrain, args.terrain)
#     if args.terrain not in ['slope', 'stair', 'stair_down', 'rough', 'gap', 'pit', 'tilt', 'crawl']:
#         print('terrain should be one of slope, stair/stair_up, stair_down, rough, gap, pit, tilt, crawl; set to stair as default')
#         args.terrain = 'stair'
#     env_cfg.terrain.terrain_proportions = {
#         'slope':      [0, 1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0],
#         'stair':      [0, 0, 1.0, 0.0, 0, 0, 0, 0, 0, 0],
#         'stair_down': [0, 0, 0.0, 1.0, 0, 0, 0, 0, 0, 0],
#         'rough':      [0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 1.0],
#         'climb':      [0, 0.1, 0.45, 0.25, 0, 0, 0, 0, 0, 0.2],
#         'gap':        [0, 0, 0.0, 0.0, 0, 1.0, 0, 0, 0, 0],
#         'pit':        [0, 0, 0.0, 0.0, 0, 0, 1.0, 0, 0, 0], 
#         'tilt':       [0, 0, 0.0, 0.0, 0, 0, 0, 1.0, 0, 0],
#         'crawl':      [0, 0, 0.0, 0.0, 0, 0, 0, 0, 1.0, 0],
#     }[args.terrain]

#     env_cfg.commands.ranges.lin_vel_x = [0.6, 0.6]
#     env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
#     env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
#     env_cfg.commands.ranges.heading = [0, 0]
#     env_cfg.commands.ranges.flat_lin_vel_x = [0.6, 0.6]
#     env_cfg.commands.ranges.flat_lin_vel_y = [0.0, 0.0]
#     env_cfg.commands.ranges.flat_ang_vel_yaw = [0.0, 0.0]

#     env_cfg.depth.use_camera = True

#     # prepare environment
#     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
#     fix_terrain_level(env, args.terrain_level)
#     env.lookat_id = min(4, env.num_envs - 1)

#     watch_id = env.lookat_id
#     print("lookat_id:", env.lookat_id)
#     print("depth_index:", env.depth_index)
#     print("depth slot:", env.depth_index_inverse[env.lookat_id])


#     print("===== DOF order from Isaac Gym =====")
#     for i, name in enumerate(env.dof_names):
#         print(i, name)

#     print("===== Wheel joints =====")
#     for i, name in enumerate(env.dof_names):
#         if "foot_joint" in name:
#             print(i, name)

#     _, _ = env.reset()
#     obs = env.get_observations()

#     # load policy
#     train_cfg.runner.resume = True
#     train_cfg.runner.load_run = args.load_run or train_cfg.runner.run_name
#     train_cfg.runner.checkpoint = -1 if args.checkpoint is None else args.checkpoint

#     ppo_runner, train_cfg = task_registry.make_wmp_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
#     policy = ppo_runner.get_inference_policy(device=env.device)

#     if EXPORT_POLICY:
#         path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
#         onnx_path = export_policy_as_onnx(
#             ppo_runner.alg.actor_critic,
#             path,
#             obs_dim=env.num_obs,
#             history_dim=ppo_runner.history_dim,
#             wm_feature_dim=ppo_runner.wm_feature_dim,
#         )
#         print('Exported WMP policy as ONNX to: ', onnx_path)

#     logger = Logger(env.dt)
#     robot_index = watch_id
#     joint_index = 1
#     stop_state_log = 100
#     stop_rew_log = env.max_episode_length + 1
#     camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
#     camera_vel = np.array([1., 1., 0.])
#     camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
#     img_idx = 0
#     video = None
#     record_cam_handle = None
#     record_width = 512
#     record_height = 512
#     record_path = os.path.abspath('record.mp4')
#     if RECORD_FRAMES:
#         if cv2 is None:
#             raise ImportError("RECORD_FRAMES=True requires opencv-python / cv2")
#         record_cam_handle = create_recording_camera(env, watch_id, record_width, record_height)
#         print(f"Recording camera attached to env {watch_id}; saving video to {record_path}")

#     # trajectory history
#     history_length = 5
#     trajectory_history = torch.zeros(size=(env.num_envs, history_length, env.num_obs -
#                                             env.privileged_dim - env.height_dim - 3), device=env.device)
#     obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
#                                         obs[:, env.privileged_dim + 9:-env.height_dim]), dim=1)
#     trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

#     # world model
#     world_model = ppo_runner._world_model.to(env.device)
#     wm_latent = None
#     wm_latent_zero_depth = None
#     wm_action = None
#     wm_is_first = torch.ones(env.num_envs, device=env.device)
#     wm_update_interval = env.cfg.depth.update_interval
#     wm_action_history = torch.zeros(size=(env.num_envs, wm_update_interval, env.num_actions),
#                                     device=env.device)
#     wm_obs = {
#         "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
#         "is_first": wm_is_first,
#     }
#     if env.cfg.depth.use_camera:
#         wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
#                                       device=world_model.device)

#     wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)
#     wm_feature_zero_depth = torch.zeros_like(wm_feature)

#     def refresh_wm_features(wm_obs, wm_latent, wm_latent_zero_depth, wm_action, force_depth=False, infos=None):
#         wm_obs_zero_depth = {
#             "prop": wm_obs["prop"],
#             "is_first": wm_obs["is_first"],
#         }
#         if env.cfg.depth.use_camera:
#             if "image" not in wm_obs:
#                 wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
#                                               device=world_model.device)
#             if force_depth and hasattr(env, "update_depth_buffer"):
#                 env.update_depth_buffer(force=True)
#             if hasattr(env, "depth_buffer") and len(env.depth_index) > 0:
#                 wm_obs["image"][env.depth_index] = env.depth_buffer[:, -2].unsqueeze(-1).to(world_model.device)
#             elif infos is not None and "depth" in infos and infos["depth"] is not None:
#                 wm_obs["image"][env.depth_index] = infos["depth"].unsqueeze(-1).to(world_model.device)
#             wm_obs_zero_depth["image"] = torch.zeros_like(wm_obs["image"])

#         with torch.no_grad():
#             wm_embed = world_model.encoder(wm_obs)
#             wm_latent, _ = world_model.dynamics.obs_step(
#                 wm_latent,
#                 wm_action,
#                 wm_embed,
#                 wm_obs["is_first"],
#                 sample=False,
#             )
#             wm_feature = world_model.dynamics.get_deter_feat(wm_latent)

#             wm_embed_zero_depth = world_model.encoder(wm_obs_zero_depth)
#             wm_latent_zero_depth, _ = world_model.dynamics.obs_step(
#                 wm_latent_zero_depth,
#                 wm_action,
#                 wm_embed_zero_depth,
#                 wm_obs["is_first"],
#                 sample=False,
#             )
#             wm_feature_zero_depth = world_model.dynamics.get_deter_feat(wm_latent_zero_depth)
#         return wm_latent, wm_latent_zero_depth, wm_feature, wm_feature_zero_depth

#     if env.cfg.depth.use_camera and hasattr(env, "update_depth_buffer"):
#         wm_latent, wm_latent_zero_depth, wm_feature, wm_feature_zero_depth = refresh_wm_features(
#             wm_obs,
#             wm_latent,
#             wm_latent_zero_depth,
#             wm_action,
#             force_depth=True,
#         )
#         wm_is_first[:] = 0

#     total_reward = 0
#     not_dones = torch.ones((env.num_envs,), device=env.device)
#     infos = {}
#     play_steps = 1 * int(env.max_episode_length) + 3
#     if RECORD_FRAMES:
#         video_duration = 20
#         play_steps = max(play_steps, int(video_duration / env.dt))
#         print(f"Gathering {play_steps} video frames")

#     warmup_steps = max(0, int(getattr(args, "play_warmup_steps", 0) or 0))
#     if warmup_steps > 0:
#         warmup_mode = getattr(args, "play_warmup_mode", "latent")
#         zero_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
#         if warmup_mode == "sim":
#             print(f"Running play warmup for {warmup_steps} zero-action physics steps")
#             for warmup_i in range(warmup_steps):
#                 obs, _, _, dones, infos, reset_env_ids, _ = env.step(zero_actions)

#                 wm_action_history = torch.concat(
#                     (wm_action_history[:, 1:], zero_actions.unsqueeze(1)), dim=1)
#                 wm_obs = {
#                     "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
#                     "is_first": wm_is_first,
#                 }
#                 if env.cfg.depth.use_camera:
#                     wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
#                                                   device=world_model.device)

#                 reset_env_ids_np = reset_env_ids.cpu().numpy()
#                 if len(reset_env_ids_np) > 0:
#                     wm_action_history[reset_env_ids_np, :] = 0
#                     wm_is_first[reset_env_ids_np] = 1

#                 wm_action = wm_action_history.flatten(1)
#                 if env.global_counter % wm_update_interval == 0:
#                     wm_latent, wm_latent_zero_depth, wm_feature, wm_feature_zero_depth = refresh_wm_features(
#                         wm_obs,
#                         wm_latent,
#                         wm_latent_zero_depth,
#                         wm_action,
#                         force_depth=True,
#                         infos=infos,
#                     )
#                     wm_is_first[:] = 0

#                 env_ids = dones.nonzero(as_tuple=False).flatten()
#                 trajectory_history[env_ids] = 0
#                 obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
#                                                     obs[:, env.privileged_dim + 9:-env.height_dim]),
#                                                    dim=1)
#                 trajectory_history = torch.concat(
#                     (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)
#         else:
#             print(f"Running play warmup for {warmup_steps} world-model latent steps")
#             wm_action = wm_action_history.flatten(1)
#             for warmup_i in range(warmup_steps):
#                 wm_latent, wm_latent_zero_depth, wm_feature, wm_feature_zero_depth = refresh_wm_features(
#                     wm_obs,
#                     wm_latent,
#                     wm_latent_zero_depth,
#                     wm_action,
#                     force_depth=True,
#                     infos=infos,
#                 )
#                 wm_is_first[:] = 0

#         print("Play warmup done; starting policy rollout")

#     for i in range(play_steps):
#         if env.global_counter % wm_update_interval == 0:
#             wm_latent, wm_latent_zero_depth, wm_feature, wm_feature_zero_depth = refresh_wm_features(
#                 wm_obs,
#                 wm_latent,
#                 wm_latent_zero_depth,
#                 wm_action,
#                 infos=infos,
#             )
#             wm_is_first[:] = 0

#         history = trajectory_history.flatten(1).to(env.device)

#         actions_wm = policy(obs.detach(), history.detach(), wm_feature.detach())
#         actions_zero_depth = policy(obs.detach(), history.detach(), wm_feature_zero_depth.detach())
#         actions_zero_wm = policy(obs.detach(), history.detach(), torch.zeros_like(wm_feature))

#         if i % 20 == 0:
#             zero_wm_diff_all = torch.mean(torch.abs(actions_wm - actions_zero_wm)).item()
#             zero_wm_diff_watch = torch.mean(torch.abs(actions_wm[watch_id] - actions_zero_wm[watch_id])).item()
#             image_diff_all = torch.mean(torch.abs(actions_wm - actions_zero_depth)).item()
#             image_diff_watch = torch.mean(torch.abs(actions_wm[watch_id] - actions_zero_depth[watch_id])).item()
#             print(
#                 "wm diff all:", zero_wm_diff_all,
#                 "watch:", zero_wm_diff_watch,
#                 "| image diff all:", image_diff_all,
#                 "watch:", image_diff_watch,
#             )

#             slot = int(env.depth_index_inverse[watch_id])
#             if hasattr(env, "depth_buffer") and slot >= 0:
#                 d = env.depth_buffer[slot, -2]
#                 infos_has_depth = "depth" in infos and infos["depth"] is not None
#                 print(
#                     "depth slot:", slot,
#                     "buffer stats:", d.min().item(), d.max().item(), d.mean().item(), d.std().item(),
#                     "infos_depth:", infos_has_depth,
#                 )
#             elif "depth" in infos and infos["depth"] is not None and slot >= 0:
#                 d = infos["depth"][slot]
#                 print("depth slot:", slot, "infos stats:", d.min().item(), d.max().item(), d.mean().item(), d.std().item())
#             else:
#                 has_depth = "depth" in infos and infos["depth"] is not None
#                 print("depth slot:", slot, "has_depth:", has_depth)

#             if getattr(args, "play_debug_metrics", False):
#                 obstacle_height, obstacle_distance, obstacle_visible = env._front_obstacle_profile()
#                 front_clearance = env._foot_clearance_above_support(env.front_foot_indices)
#                 front_impact = env._front_wheel_impact_score()
#                 rear_leg_dev = torch.sum(
#                     torch.square(
#                         env.dof_pos[:, env.rear_leg_indices]
#                         - env.default_dof_pos[:, env.rear_leg_indices]
#                     ),
#                     dim=1,
#                 )
#                 print(
#                     "metrics watch:",
#                     "obstacle_h", obstacle_height[watch_id].item(),
#                     "obstacle_d", obstacle_distance[watch_id].item(),
#                     "visible", bool(obstacle_visible[watch_id].item()),
#                     "front_clearance", front_clearance[watch_id].detach().cpu().numpy(),
#                     "front_impact", front_impact[watch_id].item(),
#                     "base_vx", env.base_lin_vel[watch_id, 0].item(),
#                     "grav_xy", env.projected_gravity[watch_id, :2].detach().cpu().numpy(),
#                     "rear_leg_dev", rear_leg_dev[watch_id].item(),
#                 )

#         actions = actions_wm

#         obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())

#         not_dones *= (~dones)
#         total_reward += torch.mean(rews * not_dones)

#         # update world model input
#         wm_action_history = torch.concat(
#             (wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
#         wm_obs = {
#             "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
#             "is_first": wm_is_first,
#         }
#         if env.cfg.depth.use_camera:
#             wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
#                                           device=world_model.device)

#         reset_env_ids = reset_env_ids.cpu().numpy()
#         if len(reset_env_ids) > 0:
#             wm_action_history[reset_env_ids, :] = 0
#             wm_is_first[reset_env_ids] = 1

#         wm_action = wm_action_history.flatten(1)

#         # process trajectory history
#         env_ids = dones.nonzero(as_tuple=False).flatten()
#         trajectory_history[env_ids] = 0
#         obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
#                                             obs[:, env.privileged_dim + 9:-env.height_dim]),
#                                            dim=1)
#         trajectory_history = torch.concat(
#             (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

#         if RECORD_FRAMES:
#             env.gym.step_graphics(env.sim)
#             env.gym.render_all_camera_sensors(env.sim)
#             img = env.gym.get_camera_image(
#                 env.sim,
#                 env.envs[watch_id],
#                 record_cam_handle,
#                 gymapi.IMAGE_COLOR,
#             )
#             img = img.reshape((record_height, record_width, 4))[:, :, :3]
#             if video is None:
#                 video = cv2.VideoWriter(
#                     record_path,
#                     cv2.VideoWriter_fourcc(*'mp4v'),
#                     int(1 / env.dt),
#                     (img.shape[1], img.shape[0]),
#                 )
#                 if not video.isOpened():
#                     raise RuntimeError(f"Failed to open video writer: {record_path}")
#             video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#             img_idx += 1
#         if MOVE_CAMERA:
#             lootat = env.root_states[watch_id, :3]
#             camara_position = lootat.detach().cpu().numpy() + [0, 1, 0]
#             env.set_camera(camara_position, lootat)

#         if i < stop_state_log:
#             logger.log_states(
#                 {
#                     'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
#                     'dof_pos': env.dof_pos[robot_index, joint_index].item(),
#                     'dof_vel': env.dof_vel[robot_index, joint_index].item(),
#                     'dof_torque': env.torques[robot_index, joint_index].item(),
#                     'command_x': env.commands[robot_index, 0].item(),
#                     'command_y': env.commands[robot_index, 1].item(),
#                     'command_yaw': env.commands[robot_index, 2].item(),
#                     'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
#                     'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
#                     'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
#                     'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
#                     'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
#                 }
#             )
#         if 0 < i < stop_rew_log:
#             if infos["episode"]:
#                 num_episodes = torch.sum(env.reset_buf).item()
#                 if num_episodes > 0:
#                     logger.log_rewards(infos["episode"], num_episodes)
#         elif i == stop_rew_log:
#             logger.print_rewards()

#     print('total reward:', total_reward)
#     if video is not None:
#         video.release()
#         print(f"Saved recording to: {record_path}")


# if __name__ == '__main__':
#     EXPORT_POLICY = True
#     RECORD_FRAMES = True
#     MOVE_CAMERA = True
#     args = get_args()
#     args.rl_device = args.sim_device
#     play(args)
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_3m_as_jit, task_registry, Logger, export_2m_as_onnx

import numpy as np
import torch
from legged_gym.utils.keyboard import handle_keyboard_events


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_restitution = False
    # env_cfg.commands.heading_command = True

    env_cfg.domain_rand.friction_range = [1.0, 1.0]
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]
    env_cfg.domain_rand.added_mass_range = [0., 0.]  # kg
    env_cfg.domain_rand.com_x_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_y_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_z_pos_range = [-0.0, 0.0]

    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = True
    # env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    # env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_motor_strength = False

    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]


    # env_cfg.terrain.mesh_type = 'plane'
    if(env_cfg.terrain.mesh_type == 'plane'):
        env_cfg.rewards.scales.feet_edge = 0
        env_cfg.rewards.scales.feet_stumble = 0


    if(args.terrain not in ['slope', 'stair', 'gap', 'climb', 'crawl', 'tilt','stone','flat','stair-down']):
        print('terrain should be one of slope, stair, gap, climb, crawl, and tilt, set to climb as default')
        args.terrain = 'climb'
    env_cfg.terrain.terrain_proportions = {
        'slope': [0, 1.0, 0.0, 0, 0, 0, 0, 0, 0,0],
        'stair': [0, 0, 1.0, 0, 0, 0, 0, 0, 0,0],
        'stair-down': [0, 0, 0, 1.0, 0, 0, 0, 0, 0,0],
        'gap': [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0,0],
        'climb': [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0,0],
        'tilt': [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0,0],
        'crawl': [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0 ,0],
        'stone': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0 ,0.],
        'flat': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1.0],        
     }[args.terrain]

    env_cfg.commands.ranges.lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [.0, .0]
    env_cfg.commands.ranges.heading = [-0., -0.]

    env_cfg.commands.ranges.flat_lin_vel_x = [-0., -0.]
    env_cfg.commands.ranges.flat_lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.flat_ang_vel_yaw = [0, 0]

    env_cfg.depth.use_camera = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy
# load policy
    train_cfg.runner.resume = True

    # 指定要加载的 run 文件夹
    train_cfg.runner.load_run = args.load_run or "WMP_titatit_quad_template_41"

    # 不传 --checkpoint 时加载该 run 的最新 checkpoint。
    train_cfg.runner.checkpoint = -1 if args.checkpoint is None else args.checkpoint

    ppo_runner, train_cfg = task_registry.make_wmp_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg
    )
    para = ppo_runner.alg.actor_critic.state_dict()
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, train_cfg.runner.load_run, 'exported', 'policies')
        export_3m_as_jit(ppo_runner.alg.actor_critic, ppo_runner._world_model, path)
        print('Exported policy as jit script to: ', path)
        export_2m_as_onnx(ppo_runner.alg.actor_critic, ppo_runner._world_model, path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    history_length = 5
    trajectory_history = torch.zeros(size=(env.num_envs, history_length, env.num_obs -
                                            env.privileged_dim - env.height_dim - 3), device = env.device)
    obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                        obs[:, env.privileged_dim + 9:-env.height_dim]), dim=1)
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

    world_model = ppo_runner._world_model.to(env.device)
    wm_latent = wm_action = None
    wm_is_first = torch.ones(env.num_envs, device=env.device)
    wm_update_interval = env.cfg.depth.update_interval
    wm_action_history = torch.zeros(size=(env.num_envs, wm_update_interval, env.num_actions),
                                    device=env.device)
    wm_obs = {
        "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
        "is_first": wm_is_first,
    }

    if (env.cfg.depth.use_camera):
        wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                      device=world_model.device)

    wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)

    total_reward = 0
    not_dones = torch.ones((env.num_envs,), device=env.device)
    num_logged_dofs = int(env.dof_pos.shape[1])

    # 用于数据实时显示
    log_f = open(os.path.join(os.path.dirname(__file__), "titatit_log.csv"), "w")
    print(
        "step,"
        + ",".join([f"dof_pos_{i}" for i in range(num_logged_dofs)])
        + ","
        + ",".join([f"dof_torque_{i}" for i in range(num_logged_dofs)])
        + ","
        + ",".join([f"contact_fz_{i}" for i in range(4)]),file=log_f,flush=True
    )

    for i in range(100000):
        # env.commands[:,:] = 0
        # env.commands[:,2] = 1
    
        if (env.global_counter % wm_update_interval == 0):
            if (env.cfg.depth.use_camera):
                wm_obs["image"][env.depth_index] = infos["depth"].unsqueeze(-1).to(world_model.device)

            wm_embed = world_model.encoder(wm_obs)
            wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=False)
            wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
            wm_is_first[:] = 0

        history = trajectory_history.flatten(1).to(env.device)
        actions = policy(obs.detach(), history.detach(), wm_feature.detach())
        # if actions[5].max() > 7:
        #     print(actions[5].max())
        obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())

        # 键盘遥控
        env_ids = torch.arange(env.num_envs, device=env.device)
        handle_keyboard_events(env, env_ids)

        # 用于数据实时显示 -1
        row = [str(i)]
        row += [str(env.dof_pos[robot_index, j].item()) for j in range(num_logged_dofs)]
        row += [str(env.torques[robot_index, j].item()) for j in range(num_logged_dofs)]
        row += [str(env.contact_forces[robot_index, env.feet_indices[j], 2].item()) for j in range(4)]
        print(",".join(row), file=log_f,flush=True)
        
        not_dones *= (~dones)
        total_reward += torch.mean(rews * not_dones)

        # update world model input
        wm_action_history = torch.concat(
            (wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
        wm_obs = {
            "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
            "is_first": wm_is_first,
        }
        if (env.cfg.depth.use_camera):
            wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                          device=world_model.device)

        reset_env_ids = reset_env_ids.cpu().numpy()
        if (len(reset_env_ids) > 0):
            wm_action_history[reset_env_ids, :] = 0
            wm_is_first[reset_env_ids] = 1

        wm_action = wm_action_history.flatten(1)
        # print(_[:,:3],_[:,236:271])

        # process trajectory history
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                            obs[:, env.privileged_dim + 9:-env.height_dim]),
                                           dim=1)
        trajectory_history = torch.concat(
            (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            lootat = env.root_states[8, :3]
            camara_position = lootat.detach().cpu().numpy() + [0, 1, 0]
            env.set_camera(camara_position, lootat)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

    print('total reward:', total_reward)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    args.rl_device = args.sim_device
    play(args)
