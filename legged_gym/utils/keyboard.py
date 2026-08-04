from isaacgym import gymapi
from isaacgym import gymtorch
import torch


def handle_keyboard_events(env, env_ids):
    for evt in env.gym.query_viewer_action_events(env.viewer):
        need_reset = False
        if evt.value > 0:
            if evt.action == "Suddenly_appear":
                x, y, z = env._get_height_ahead()
                env.root_states[:, 0] = x
                env.root_states[:, 1] = y
                env.root_states[:, 2] = z + 0.65
                need_reset = True

            elif evt.action == "Push_robots":
                max_vel = 2.5
                from isaacgym.torch_utils import torch_rand_float

                push_vel = torch.tensor([max_vel, max_vel], device=env.device)
                random_axis = torch.randint(0, 2, (1,)).item()
                random_vel = torch_rand_float(-max_vel, max_vel, (1, 1), device=env.device).item()
                push_vel[random_axis] = random_vel
                env.root_states[0, 7:9] = push_vel
                print(f"Applied random push: X={push_vel[0]:.2f} m/s, Y={push_vel[1]:.2f} m/s")

            elif evt.action == "reset_robots":
                last_root_states = env.root_states.clone()
                env.root_states[:, :] = env.base_init_state
                env.root_states[:, :2] += last_root_states[:, :2]
                env.root_states[:, 2:3] = last_root_states[:, 2:3] + 0.3
                need_reset = True

            if evt.action == "left" and evt.value > 0:
                env.commands[:, 1] = 1
            elif evt.action == "right" and evt.value > 0:
                env.commands[:, 1] = -1
            else:
                env.commands[:, 1] = 0

            if evt.action == "up" and evt.value > 0:
                env.commands[:, 0] = 1
            elif evt.action == "down" and evt.value > 0:
                env.commands[:, 0] = -1
            else:
                env.commands[:, 0] = 0

            if evt.action == "yaw_left" and evt.value > 0:
                env.commands[:, 2] = 1
            elif evt.action == "yaw_right" and evt.value > 0:
                env.commands[:, 2] = -1
            else:
                env.commands[:, 2] = 0

            if evt.action == "stand_z" and evt.value > 0:
                env.commands[:, 4] = 1 - env.commands[:, 4]

            if need_reset:
                env_ids_int32 = env_ids.to(dtype=torch.int32)
                env.gym.set_actor_root_state_tensor_indexed(
                    env.sim,
                    gymtorch.unwrap_tensor(env.root_states),
                    gymtorch.unwrap_tensor(env_ids_int32),
                    len(env_ids_int32),
                )
                env.dof_pos[:] = env.default_dof_pos
                env.dof_vel[:] = 0.0
                env.gym.set_dof_state_tensor_indexed(
                    env.sim,
                    gymtorch.unwrap_tensor(env.dof_state),
                    gymtorch.unwrap_tensor(env_ids_int32),
                    len(env_ids_int32),
                )
