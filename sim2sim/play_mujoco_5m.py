import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

import pygame
from threading import Thread
import os, sys, cv2, glfw
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.keyboard_controller import KeyboardController
from mjc_utils import start_joystick_thread, DepthViewer

Command_Generator = KeyboardController(max_vel = 1)
Command_Generator.start_listening()
x_vel_cmd, y_vel_cmd, yaw_vel_cmd, height_vel_cmd = 0.0, 0.0, 0.0, 0.0
joystick_use = False
joystick_opened = False
reset_pos = False      

from timeit import default_timer as timer
from contextlib import contextmanager

# 定义 time-block（毫秒单位）
@contextmanager
def time_block(name="block"):
    start = timer()
    yield
    end = timer()
    elapsed_ms = (end - start) * 1000
    print(f"{name} 执行时间: {elapsed_ms:.3f} ms")


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def apply_action_filter(action_mj, previous_action_mj, filter_cfg):
    if not filter_cfg.get("enabled", False):
        return action_mj
    w_last = filter_cfg.get("last_action_weight", 0.2)
    w_curr = filter_cfg.get("current_action_weight", 0.8)
    return w_last * previous_action_mj + w_curr * action_mj


class SimulationRunner:
    def __init__(self, config_file, scene_path,jit_load_path):
        self.device = "cuda" #"cpu"
        self.load_config(config_file, scene_path, jit_load_path)
        self.init_context()
        self.load_model()
        self.load_policy()

        # 启动手柄线程
        if joystick_use: self.joystick_thread = start_joystick_thread()

    # ------------------------- 配置加载 -------------------------
    def load_config(self, config_file, scene_path, jit_load_path):
        self.xml_path = scene_path
        self.jit_load_path = jit_load_path
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.simulation_duration = config["simulation_duration"]
        self.simulation_dt = config["simulation_dt"]
        self.control_decimation = config["control_decimation"]

        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)

        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.ang_vel_scale = config["ang_vel_scale"]
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]
        self.action_scale = config["action_scale"]
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        self.num_actions = config["num_actions"]
        self.num_obs = config.get("num_obs", 54)
        self.wm_prop_dim = config.get("wm_prop_dim", 57)
        self.clip_actions = float(config.get("clip_actions", 7.0))

        self.wheel_kp = float(config.get("wheel_kp", 0.4))
        self.wheel_kd_scale = float(config.get("wheel_kd_scale", 0.4))
        self.wheel_kp_scale = float(config.get("wheel_kp_scale", self.kps[0] * self.wheel_kp))
        self.wheel_kd_abs = float(config.get("wheel_kd_abs", self.kds[0] * self.wheel_kd_scale))
        self.hip_scale_reduction = float(config.get("hip_scale_reduction", 1.0))
        self.wheel_indices = config.get("wheel_indices", [3, 7, 11, 15])
        self.hip_indices = config.get("hip_indices", [0, 4, 8, 12])
        self.action_filter_cfg = config.get("action_filter", {"enabled": True, "last_action_weight": 0.2, "current_action_weight": 0.8})

        # 初始 command
        self.cmd = np.array(config["cmd_init"], dtype=np.float32)

    # ------------------------- 上下文变量 -------------------------
    def init_context(self):
        self.num_hist = 5
        self.counter = 0
        self.wm_update_interval = 5
        self.policy_step_counter = 0  

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.previous_action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.policy_prop = np.zeros(self.num_obs, dtype=np.float32)
        self.wm_prop = np.zeros(self.wm_prop_dim, dtype=np.float32)
        self.obs_history = np.zeros((self.num_hist, self.num_obs), dtype=np.float32)
        self.wm_image = torch.zeros((1, 64, 64, 1), device=self.device)
        self.prev_wm_image = None
        
        self.wm_is_first = torch.ones(1, device=self.device)
        self.wm_latent = torch.zeros(1, 32, 32, device=self.device)
        self.wm_stoch = torch.zeros(1, 32, 32, device=self.device)
        self.wm_deter = torch.zeros(1, 512, device=self.device)
        self.wm_feature = torch.zeros(1, 512, device=self.device)

        self.wm_action_history = torch.zeros(1, self.wm_update_interval, self.num_actions,device=self.device)
        self.wm_action = torch.zeros(1, self.wm_update_interval * self.num_actions, device=self.device)
        self.torque_limits = np.array([60., 60., 90., 60., 60., 90., 60., 60., 90., 60., 60., 90.])


    # ------------------------- Mujoco 模型 -------------------------
    def load_model(self):
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt
        self.renderer = mujoco.Renderer(self.m, width=64, height=64)
        
    # ------------------------- Load jit Model -------------------------
    def load_policy(self):
        self.jit_load_path
        self.wm_encoder = torch.jit.load(os.path.join(self.jit_load_path, 'wm_enc.pt'), map_location=self.device)
        self.hist_encoder = torch.jit.load(os.path.join(self.jit_load_path, 'hist_enc.pt'), map_location=self.device)
        self.actor = torch.jit.load(os.path.join(self.jit_load_path, 'actor.pt'), map_location=self.device)
        
        # wm_model.eval()
        self.hist_encoder.eval()
        self.actor.eval()

        obs_shape = {'prop': (33,), 'image': (64,64,1)}
        from dreamer import networks
        self.encoder = networks.MultiEncoder(obs_shape,'.*','image','SiLU',True,32,4,4,5,1024,True,True)
        self.dynamics = networks.RSSM(32,512,512,1,32,'SiLU',True,'none','sigmoid2',0.1,0.01,'learned',60,5120,self.device)
        # 原始 state_dict
        wm_model = torch.load('/data/train/wmp/WMP-panda3/logs/panda3_amp_example/Feb24_11-31-32_WMP/model_21000.pt', map_location=self.device)
        full_dict = wm_model['world_model_dict']

        # 过滤 encoder 和 dynamics 的参数（去掉前缀以匹配子模块）
        encoder_dict = {
            k.replace('encoder.', '', 1): v
            for k, v in full_dict.items()
            if k.startswith('encoder.')
        }

        dynamics_dict = {
            k.replace('dynamics.', '', 1): v
            for k, v in full_dict.items()
            if k.startswith('dynamics.')
        }

        # 加载参数到各自模块
        self.encoder.load_state_dict(encoder_dict, strict=False)
        self.dynamics.load_state_dict(dynamics_dict, strict=False)

        self.encoder.to(self.device)
        self.dynamics.to(self.device)
        self.encoder.eval()
        self.dynamics.eval()


    # ---------------- OBS ----------------

    def build_observation(self):
        d = self.d
        qj = ((d.qpos[7:] - self.default_angles) * self.dof_pos_scale).astype(np.float32, copy=False)
        dqj = (d.qvel[6:] * self.dof_vel_scale).astype(np.float32, copy=False)
        quat = d.qpos[3:7]
        
        from utils.math import quat_rotate_inverse
        omega = (quat_rotate_inverse(quat, d.qvel[3:6]) * self.ang_vel_scale).astype(np.float32, copy=False)
        
        gravity = get_gravity_orientation(quat)
        self.cmd = (Command_Generator.get_velocities().astype(np.float32) * self.cmd_scale).astype(np.float32, copy=False)

        self.policy_prop[:3] = omega
        self.policy_prop[3:6] = gravity
        self.policy_prop[6:22] = qj
        self.policy_prop[22:38] = dqj
        self.policy_prop[38:54] = self.action

        self.wm_prop[:3] = omega
        self.wm_prop[3:6] = gravity
        self.wm_prop[6:9] = self.cmd[:3]
        self.wm_prop[9:25] = qj
        self.wm_prop[25:41] = dqj
        self.wm_prop[41:57] = self.action

    # ---------------- POLICY ----------------
    def image_render(self):

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.d, camera="depth_cam")
        @timer
        def test():
            depth = self.renderer.render()
            return depth
        depth = test()
        self.renderer.disable_depth_rendering()
        depth_vis = torch.from_numpy(depth).to(self.device)
        depth_vis = torch.clamp(depth_vis, 0, 2) / 2.0 - 0.5
        # self.wm_image = depth_vis.unsqueeze(0).unsqueeze(-1)
        self.depth_viewer.update(depth_vis.cpu().numpy() + 0.5)
        return depth_vis.unsqueeze(0).unsqueeze(-1)

    def policy_inference(self):
        input_wm_prop = torch.tensor(self.wm_prop, dtype=torch.float32, device=self.device).unsqueeze(0)
        history_tensor = torch.tensor(self.obs_history, dtype=torch.float32, device=self.device).unsqueeze(0)
        cmd_tensor = torch.tensor(self.cmd[:3], dtype=torch.float32, device=self.device).unsqueeze(0)

        self.wm_obs =  {
            "prop": input_wm_prop,
            "is_first": self.wm_is_first}

            
        # ===== 每 5 次 policy 更新一次 World Model =====
        if self.policy_step_counter % self.wm_update_interval == 0:  # 100ms
            # 先刷新并显示图片
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.d, camera="depth_cam")
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()

            depth_vis = np.clip(depth, 0, 2) / 2.0 - 0.5
            current_wm_image = torch.from_numpy(depth_vis).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            self.depth_viewer.update(depth_vis + 0.5)
            wm_input_image = current_wm_image if self.prev_wm_image is None else self.prev_wm_image
            self.wm_obs["image"] = wm_input_image

            # world model
            with time_block("wm推理"):
                self.wm_embed = self.encoder(self.wm_obs)
                self.wm_latent, _ = self.dynamics.obs_step(self.wm_latent, self.wm_action, self.wm_embed, self.wm_obs["is_first"], sample=True)
                self.wm_feature = self.dynamics.get_deter_feat(self.wm_latent)
                self.wm_is_first[:] = 0
                self.prev_wm_image = current_wm_image

        # ===== Policy 推理 =====
        history_flat = history_tensor.flatten(1)
        # actor_model
        with time_block("policy推理"):
            latent_vector = self.hist_encoder(history_flat)
            wm_latent_vector = self.wm_encoder(self.wm_feature)
            concat_observations = torch.concat((latent_vector, torch.from_numpy(self.cmd).float().unsqueeze(0).to(self.device), wm_latent_vector),dim=-1)
            action = self.actor(concat_observations)

        # ===== 更新 WM action history =====
        action_tensor = action.unsqueeze(0)
        self.wm_action_history = torch.cat((self.wm_action_history[:, 1:], action_tensor), dim=1)
        self.wm_action = self.wm_action_history.flatten(1)

        self.policy_step_counter += 1
        return action.detach().cpu().numpy()

    # ------------------------- 主循环 -------------------------
    def run(self):
        self.depth_viewer = DepthViewer(scale=6)
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < self.simulation_duration:
                step_start = time.time()
                if self.counter % self.control_decimation == 0: # 20ms
                    self.build_observation()
                    raw_action = np.clip(self.policy_inference(), -7.0, 7.0).astype(np.float32, copy=False)
                    filtered_action = apply_action_filter(
                        raw_action,
                        self.previous_action,
                        self.action_filter_cfg,
                    )
                    self.action = filtered_action.copy()
                    self.action[self.hip_indices] *= self.hip_scale_reduction
                    self.previous_action[:] = self.action
                    self.obs_history[:-1] = self.obs_history[1:]
                    self.obs_history[-1] = self.policy_prop
                self.counter += 1

                # ===== PD 控制 =====
                q = np.asarray(self.d.qpos[7:7 + self.num_actions], dtype=np.float32)
                dq = np.asarray(self.d.qvel[6:6 + self.num_actions], dtype=np.float32)
                action_scaled = self.action * self.action_scale
                
                tau = (action_scaled + self.default_angles - q) * self.kps - dq * self.kds
                
                wheel = self.wheel_indices
                tau[wheel] = (
                    self.wheel_kp_scale * action_scaled[wheel]
                    - self.wheel_kd_abs * dq[wheel]
                )

                tau_clipped = np.clip(tau, -self.torque_limits, self.torque_limits)
                self.d.ctrl[:] = tau_clipped
                mujoco.mj_step(self.m, self.d)  # 5ms

                viewer.sync()
                dt = self.m.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)

# ------------------------- 主入口 -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name")
    parser.add_argument("scene_path", type=str, help="config file name")
    parser.add_argument("jit_load_path", type=str, help="config file name")
    args = parser.parse_args()
    
    sim = SimulationRunner(args.config_file, args.scene_path, args.jit_load_path)
    sim.run()