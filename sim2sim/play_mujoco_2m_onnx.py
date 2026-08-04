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

Command_Generator = KeyboardController(max_vel=1.0, max_yaw_vel=1.4)
Command_Generator.start_listening()
print("Keyboard control: ArrowUp/Down=forward/back, ArrowLeft/Right=yaw, Q/E=left/right strafe, Space=scram, F1=quit listener")
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
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def quat_rotate_inverse(q, v):
    w, x, y, z = q
    q_vec = np.array([x, y, z], dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * (2.0 * w)
    c = q_vec * (2.0 * np.dot(q_vec, v))
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def apply_action_filter(action_mj, previous_action_mj, filter_cfg):
    if not filter_cfg.get("enabled", False):
        return action_mj.astype(np.float32, copy=True)
    return (
        previous_action_mj * float(filter_cfg.get("last_action_weight", 0.2))
        + action_mj * float(filter_cfg.get("current_action_weight", 0.8))
    ).astype(np.float32)


def depth_image_preprocessing_numpy(depth_image: np.ndarray,
                                    near_plane: float = 0.0,
                                    far_plane: float = 2.0,
                                    depth_scale: float = 1.0) -> np.ndarray:
    """
    对齐 Gym/Warp 训练侧的深度预处理逻辑：
    - abs(depth) < near_plane / depth_scale -> 0
    - abs(depth) > far_plane / depth_scale  -> far_plane / depth_scale

    这里默认 depth_scale=1.0，表示深度单位已经是米。
    """
    depth = depth_image.copy()

    near_threshold = near_plane / depth_scale
    far_threshold = far_plane / depth_scale

    near_mask = np.abs(depth) < near_threshold
    far_mask = np.abs(depth) > far_threshold

    depth[near_mask] = 0.0
    depth[far_mask] = far_threshold

    if np.isnan(depth).any() or np.isinf(depth).any():
        raise RuntimeError("nan or inf of depth image detected!")

    return depth


class SimulationRunner:
    def __init__(self, config_file, scene_path, jit_load_path, debug_perception=False, blind=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug_perception = debug_perception
        self.blind = blind
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
        if self.kps.size == 1:
            self.kps = np.full(int(config["num_actions"]), float(self.kps[0]), dtype=np.float32)
        if self.kds.size == 1:
            self.kds = np.full(int(config["num_actions"]), float(self.kds[0]), dtype=np.float32)

        self.default_angles_model = np.array(config["default_angles"], dtype=np.float32)
        self.ang_vel_scale = config["ang_vel_scale"]
        self.lin_vel_scale = config.get("lin_vel_scale", 2.0)
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]
        self.action_scale = np.array(config["action_scale"], dtype=np.float32)
        if self.action_scale.ndim == 0:
            self.action_scale = np.full(int(config["num_actions"]), float(self.action_scale), dtype=np.float32)
        self.wheel_kp = float(config.get("wheel_kp", 0.4))
        self.wheel_kd_scale = float(config.get("wheel_kd_scale", 0.4))
        self.wheel_kp_scale = float(config.get("wheel_kp_scale", self.kps[0] * self.wheel_kp))
        self.wheel_kd_abs = float(config.get("wheel_kd_abs", self.kds[0] * self.wheel_kd_scale))
        self.hip_scale_reduction = float(config.get("hip_scale_reduction", 1.0))
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        self.num_actions = config["num_actions"]
        self.num_obs = int(config.get("num_obs", 54))
        self.wm_prop_dim = int(config.get("wm_prop_dim", 41))
        self.num_hist = int(config.get("history_len", 5))
        self.wheel_indices_model = np.array(config.get("wheel_indices", [3, 7, 11, 15]), dtype=np.int64)
        self.hip_indices_model = np.array(config.get("hip_indices", [0, 4, 8, 12]), dtype=np.int64)
        self.clip_actions = float(config.get("clip_actions", 8))
        self.action_filter_cfg = config.get("action_filter", {"enabled": True, "last_action_weight": 0.2, "current_action_weight": 0.8})

        self.cmd = np.array(config["cmd_init"], dtype=np.float32)
        self.model_joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint",
        ]
        keyboard_cfg = config.get("keyboard", {})
        Command_Generator.max_vel = float(keyboard_cfg.get("max_vel", Command_Generator.max_vel))
        Command_Generator.max_yaw_vel = float(keyboard_cfg.get("max_yaw_vel", Command_Generator.max_yaw_vel))

        # ===== 对齐训练侧 Titatit WMP depth 配置 =====
        depth_cfg = config.get("depth", {})
        self.depth_camera_name = str(depth_cfg.get("camera_name", "depth_camera"))
        self.depth_original_w, self.depth_original_h = [
            int(v) for v in depth_cfg.get("original", [64, 64])
        ]
        self.depth_resized_w, self.depth_resized_h = [
            int(v) for v in depth_cfg.get("resized", [64, 64])
        ]
        self.depth_fovy = float(depth_cfg.get("horizontal_fov", 58))
        self.depth_near = float(depth_cfg.get("near_clip", 0.0))
        self.depth_far = float(depth_cfg.get("far_clip", 2.0))
        self.depth_scale = float(depth_cfg.get("scale", 1.0))
        self.depth_update_interval = int(depth_cfg.get("update_interval", 5))
        self.depth_hide_robot = bool(depth_cfg.get("hide_robot", True))

    # ------------------------- 上下文变量 -------------------------
    def init_context(self):
        self.counter = 0
        self.wm_update_interval = self.depth_update_interval
        self.policy_step_counter = 0

        self.action_model = np.zeros(self.num_actions, dtype=np.float32)
        self.previous_action_model = np.zeros(self.num_actions, dtype=np.float32)
        self.previous_action_mj = np.zeros(self.num_actions, dtype=np.float32)
        self.action_mj = np.zeros(self.num_actions, dtype=np.float32)
        self.raw_policy_action_model = np.zeros(self.num_actions, dtype=np.float32)
        self.last_wm_ms = 0.0
        self.last_policy_ms = 0.0
        self.last_wm_norm = 0.0
        self.last_act_diff_mean = 0.0
        self.last_act_diff_max = 0.0
        self.last_depth_stats = {
            "pre_raw_min": 0.0,
            "pre_raw_max": 0.0,
            "pre_raw_std": 0.0,
            "raw_min": 0.0,
            "raw_max": 0.0,
            "norm_mean": 0.0,
            "norm_std": 0.0,
            "base_z": 0.0,
            "cam_forward_z": 0.0,
        }
        self.last_status_print_time = 0.0
        self.target_dof_pos = self.default_angles_model.copy()
        self.policy_prop = np.zeros(self.num_obs, dtype=np.float32)
        self.wm_prop = np.zeros(self.wm_prop_dim, dtype=np.float32)
        self.obs_history = np.zeros((self.num_hist, self.num_obs), dtype=np.float32)

        # ===== 对齐训练侧 resized=(64,64) =====
        self.wm_image = torch.zeros(
            (1, self.depth_resized_h, self.depth_resized_w, 1),
            device=self.device
        )
        self.prev_wm_image = None

        self.wm_is_first = torch.ones(1, device=self.device)
        self.wm_logit = torch.zeros(1, 32, 32, device=self.device)
        self.wm_stoch = torch.zeros(1, 32, 32, device=self.device)
        self.wm_deter = torch.zeros(1, 512, device=self.device)
        self.wm_feature = torch.zeros(1, 512, device=self.device)

        self.wm_action_history = torch.zeros(
            1, self.wm_update_interval, self.num_actions, device=self.device
        )
        self.wm_action = torch.zeros(
            1, self.wm_update_interval * self.num_actions, device=self.device
        )
        self.torque_limits_model = np.full(self.num_actions, 55.0, dtype=np.float32)
        self.torque_limits_model[self.wheel_indices_model] = 10.0

        self.depth_points = torch.zeros(self.depth_resized_h, self.depth_resized_w, 3)

    # ------------------------- Mujoco 模型 -------------------------
    def load_model(self):
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt
        self._init_joint_order_mapping()

        # ===== 按训练侧 original=(64,64) 渲染 =====
        self.renderer = mujoco.Renderer(
            self.m,
            width=self.depth_original_w,
            height=self.depth_original_h
        )
        self._setup_depth_scene_option()

        self.initial_qpos = self.m.qpos0.copy()
        self.initial_qpos[7:7 + self.num_actions] = self.default_angles_mj
        self.base_reset_height = 0.18

    def _init_joint_order_mapping(self):
        mj_joint_names = []
        for joint_id in range(self.m.njnt):
            if self.m.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
                mj_joint_names.append(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        if mj_joint_names != [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
        ]:
            print("MuJoCo joint order:", mj_joint_names)
        name_to_mj = {name: idx for idx, name in enumerate(mj_joint_names)}
        self.idx_model_to_mj = np.array([name_to_mj[name] for name in self.model_joint_names], dtype=np.int64)
        self.idx_mj_to_model = np.empty_like(self.idx_model_to_mj)
        self.idx_mj_to_model[self.idx_model_to_mj] = np.arange(self.num_actions, dtype=np.int64)
        self.default_angles_mj = self.default_angles_model[self.idx_mj_to_model]
        self.kps_model = self.kps
        self.kds_model = self.kds

    def _setup_depth_scene_option(self):
        self.depth_scene_option = None
        if not self.depth_hide_robot:
            return

        base_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_body_id < 0:
            return

        def is_robot_body(body_id):
            while body_id > 0:
                if body_id == base_body_id:
                    return True
                body_id = self.m.body_parentid[body_id]
            return False

        # Training Warp depth raycasts terrain only. Hide robot self-geometry from
        # the MuJoCo depth camera while keeping world terrain/obstacle geoms.
        for geom_id in range(self.m.ngeom):
            if is_robot_body(int(self.m.geom_bodyid[geom_id])):
                self.m.geom_group[geom_id] = 2

        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[0] = 1
        self.depth_scene_option = option

    # ------------------------- Load jit Model -------------------------
    def load_policy(self):
        import onnxruntime as ort
        wm_int8_path = os.path.join(self.jit_load_path, "world_model_int8.onnx")
        wm_fp32_path = os.path.join(self.jit_load_path, "world_model.onnx")
        wm_path = wm_int8_path if os.path.exists(wm_int8_path) else wm_fp32_path
        self.world_model = ort.InferenceSession(
            wm_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.policy = ort.InferenceSession(
            os.path.join(self.jit_load_path, "policy.onnx"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._validate_onnx_shapes()
        print(f"World model ONNX: {wm_path}")
        print(f"Policy ONNX: {os.path.join(self.jit_load_path, 'policy.onnx')}")
        print("ONNX models loaded.")
        self.reset_simulation()

    def _validate_onnx_shapes(self):
        world_inputs = {item.name: item for item in self.world_model.get_inputs()}
        policy_inputs = {item.name: item for item in self.policy.get_inputs()}

        def check_feature_dim(inputs, name, expected, model_name):
            item = inputs.get(name)
            if item is None:
                raise RuntimeError(f"{model_name} 缺少输入: {name}")
            if len(item.shape) < 2:
                raise RuntimeError(f"{model_name} 输入 {name} 维度异常: {item.shape}")
            actual = item.shape[1]
            if isinstance(actual, int) and actual > 0 and actual != expected:
                raise RuntimeError(
                    f"{model_name} 输入 {name}={actual}，但当前严格四足模板配置要求 {expected}"
                )

        check_feature_dim(world_inputs, "prop", self.wm_prop_dim, "world model")
        check_feature_dim(
            policy_inputs, "history", self.num_obs * self.num_hist, "policy"
        )
        check_feature_dim(policy_inputs, "command", 3, "policy")

        action_input = world_inputs.get("action")
        if action_input is not None and len(action_input.shape) >= 2:
            action_dim = action_input.shape[1]
            if isinstance(action_dim, int) and action_dim > 0:
                expected_interval = action_dim // self.num_actions
                if action_dim % self.num_actions != 0:
                    raise RuntimeError(
                        f"world model action 输入维度异常: {action_dim} 不能整除 num_actions={self.num_actions}"
                    )
                if expected_interval != self.wm_update_interval:
                    raise RuntimeError(
                        "sim2sim 配置与导出的 world model 不一致: "
                        f"config depth.update_interval={self.wm_update_interval}, "
                        f"但 ONNX action 维度={action_dim}, 对应训练 update_interval={expected_interval}. "
                        "请把 sim2sim/titatit.yml 里的 depth.update_interval 改回训练导出时的值。"
                    )

    def reset_runtime_buffers(self):
        self.counter = 0
        self.policy_step_counter = 0
        self.action_model.fill(0.0)
        self.previous_action_model.fill(0.0)
        self.previous_action_mj.fill(0.0)
        self.action_mj.fill(0.0)
        self.raw_policy_action_model.fill(0.0)
        self.last_wm_ms = 0.0
        self.last_policy_ms = 0.0
        self.last_status_print_time = 0.0
        self.target_dof_pos = self.default_angles_model.copy()
        self.policy_prop.fill(0.0)
        self.wm_prop.fill(0.0)
        self.obs_history.fill(0.0)
        self.prev_wm_image = None

        with torch.inference_mode():
            self.wm_is_first.fill_(1.0)
            self.wm_logit.zero_()
            self.wm_stoch.zero_()
            self.wm_deter.zero_()
            self.wm_feature.zero_()
            self.wm_image.zero_()
            self.wm_action_history.zero_()
            self.wm_action.zero_()

    def reset_simulation(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.initial_qpos
        self.d.qpos[0:3] = np.array([0.0, 0.0, 0.44], dtype=np.float32)
        self.d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.d.qpos[7:7 + self.num_actions] = self.default_angles_mj
        self.d.qvel[:] = 0.0
        self.d.ctrl[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.reset_runtime_buffers()

    def should_reset(self):
        global reset_pos
        if reset_pos:
            reset_pos = False
            return True

        gravity = get_gravity_orientation(self.d.qpos[3:7])
        return gravity[2] > 0.0 or self.d.qpos[2] < self.base_reset_height

    # ------------------------- 构造 obs -------------------------
    def build_observation(self):
        d = self.d
        q_mj = np.asarray(d.qpos[7:7 + self.num_actions], dtype=np.float32)
        dq_mj = np.asarray(d.qvel[6:6 + self.num_actions], dtype=np.float32)
        q_model = q_mj[self.idx_model_to_mj]
        dq_model = dq_mj[self.idx_model_to_mj]
        dof_pos_obs = q_model - self.default_angles_model
        dof_pos_obs[self.wheel_indices_model] = 0.0
        qj = (dof_pos_obs * self.dof_pos_scale).astype(np.float32, copy=False)
        dqj = (dq_model * self.dof_vel_scale).astype(np.float32, copy=False)
        quat = d.qpos[3:7]
        omega = (quat_rotate_inverse(quat, d.qvel[3:6]) * self.ang_vel_scale).astype(np.float32, copy=False)
        gravity = get_gravity_orientation(quat)
        self.cmd = (Command_Generator.get_velocities().astype(np.float32) * self.cmd_scale).astype(np.float32, copy=False)

        self.policy_prop[:3] = omega
        self.policy_prop[3:6] = gravity
        self.policy_prop[6:22] = qj
        self.policy_prop[22:38] = dqj
        self.policy_prop[38:54] = self.action_model

        self.wm_prop[:3] = omega
        self.wm_prop[3:6] = gravity
        self.wm_prop[6:9] = self.cmd[:3]
        self.wm_prop[9:25] = qj
        self.wm_prop[25:41] = dqj

        if self.policy_step_counter == 0:
            self.obs_history[:] = 0.0
            self.obs_history[-1] = self.policy_prop
        else:
            self.obs_history[:-1] = self.obs_history[1:]
            self.obs_history[-1] = self.policy_prop


    def depth_to_point(self,depth,fovy,width,height,cam_pos,cam_mat):
        """
        depth: 已处理好的线性深度（不是Z-buffer）
        返回: (H, W, 3)
        """

        # -------- 相机内参 --------
        f = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        fx = fy = f
        cx = width / 2
        cy = height / 2

        # -------- 像素网格 --------
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))

        z = -depth  
        x = (xs - cx) * depth / fx
        y = -(ys - cy) * depth / fy

        points = np.stack([x, y, z], axis=-1)

        # -------- 转世界坐标 --------
        H, W, _ = points.shape
        points_flat = points.reshape(-1, 3)

        # R @ p + t
        points_world = (cam_mat @ points_flat.T).T + cam_pos

        return points_world.reshape(H, W, 3)

    # ------------------------- 深度图采集与预处理 -------------------------
    def get_depth_image_aligned(self) -> np.ndarray:
        """
        输出 shape: (64, 64)
        处理流程对齐 Gym/Warp:
        1. 按 original=(64,64) 渲染
        2. near/far 预处理
        3. resize 到 resized=(64,64)
        4. 归一化到 [-0.5, 0.5]
        """
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(
            self.d,
            camera=self.depth_camera_name,
            scene_option=self.depth_scene_option,
        )
        raw_depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        # 1) 预处理：near/far 截断
        depth = depth_image_preprocessing_numpy(
            raw_depth,
            near_plane=self.depth_near,
            far_plane=self.depth_far,
            depth_scale=self.depth_scale
        )

        # 2) resize 到训练输入尺寸
        depth = cv2.resize(
            depth,
            (self.depth_resized_w, self.depth_resized_h),
            interpolation=cv2.INTER_NEAREST
        )

        cam_id = mujoco.mj_name2id(
            self.m,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.depth_camera_name
        )
        cam_pos = self.d.cam_xpos[cam_id]               # ✅ 动态位置
        cam_mat = self.d.cam_xmat[cam_id].reshape(3, 3)   # ✅ 动态旋转
        cam_forward = -cam_mat[:, 2]

        self.depth_points = self.depth_to_point(
            depth,
            fovy=self.depth_fovy,
            width=depth.shape[1],
            height=depth.shape[0],
            cam_pos=cam_pos,
            cam_mat = cam_mat
        )

        # 3) 归一化到 [-0.5, 0.5]
        depth_vis = depth / self.depth_far - 0.5
        depth_vis = np.clip(depth_vis, -0.5, 0.5).astype(np.float32)
        self.last_depth_stats = {
            "pre_raw_min": float(np.min(raw_depth)),
            "pre_raw_max": float(np.max(raw_depth)),
            "pre_raw_std": float(np.std(raw_depth)),
            "raw_min": float(np.min(depth)),
            "raw_max": float(np.max(depth)),
            "norm_mean": float(np.mean(depth_vis)),
            "norm_std": float(np.std(depth_vis)),
            "base_z": float(self.d.qpos[2]),
            "cam_forward_z": float(cam_forward[2]),
        }

        return depth_vis

    # ------------------------- Policy 推理 -------------------------
    def policy_inference(self):
        input_wm_prop = torch.from_numpy(self.wm_prop).to(self.device, dtype=torch.float32).unsqueeze(0)
        history_tensor = torch.from_numpy(self.obs_history).to(self.device, dtype=torch.float32).unsqueeze(0)

        with torch.inference_mode():
            # ===== 每 5 次 policy 更新一次 World Model =====
            if self.policy_step_counter % self.wm_update_interval == 0:
                with time_block("图像加载与归一化"):
                    depth_vis = self.get_depth_image_aligned()

                current_wm_image = torch.from_numpy(depth_vis).to(
                    self.device, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(-1)

                # 可视化时转到 [0,1]
                self.depth_viewer.update(np.clip(depth_vis + 0.5, 0.0, 1.0))
                self.start_time = time.time()

                wm_input_image = current_wm_image if self.prev_wm_image is None else self.prev_wm_image
                self.wm_image = wm_input_image

                wm_start = time.time()
                with time_block("wm推理"):
                    wm_inputs = {
                        "prop": input_wm_prop.cpu().numpy(),
                        "img": self.wm_image.cpu().numpy(),
                        "logit": self.wm_logit.cpu().numpy(),
                        "stoch": self.wm_stoch.cpu().numpy(),
                        "deter": self.wm_deter.cpu().numpy(),
                        "action": self.wm_action.cpu().numpy(),
                        "is_first": self.wm_is_first.cpu().numpy(),
                    }
                    wm_outputs = self.world_model.run(None, wm_inputs)
                    self.wm_logit = torch.tensor(wm_outputs[0]).to(self.device)
                    self.wm_stoch = torch.tensor(wm_outputs[1]).to(self.device)
                    self.wm_deter = torch.tensor(wm_outputs[2]).to(self.device)
                    self.wm_feature = torch.tensor(wm_outputs[3]).to(self.device)
                self.last_wm_ms = (time.time() - wm_start) * 1000.0

                self.wm_is_first[:] = 0
                self.prev_wm_image = current_wm_image

            # ===== Policy 推理 =====
            history_flat = history_tensor.flatten(1)
            policy_start = time.time()
            with time_block("policy推理"):
                policy_inputs = {
                    "command": self.cmd[None, :],
                    "history": history_flat.detach().cpu().numpy(),
                    "wm_feature": self.wm_feature.detach().cpu().numpy(),
                }
                outputs = self.policy.run(None, policy_inputs)
                action = outputs[0]

                if self.debug_perception or self.blind:
                    blind_inputs = dict(policy_inputs)
                    blind_inputs["wm_feature"] = np.zeros_like(policy_inputs["wm_feature"])
                    blind_action = self.policy.run(None, blind_inputs)[0]
                    diff = action - blind_action
                    self.last_wm_norm = float(np.linalg.norm(policy_inputs["wm_feature"]))
                    self.last_act_diff_mean = float(np.mean(np.abs(diff)))
                    self.last_act_diff_max = float(np.max(np.abs(diff)))
                    if self.blind:
                        action = blind_action
            self.last_policy_ms = (time.time() - policy_start) * 1000.0

            action = action.squeeze(0)

        self.policy_step_counter += 1
        return action

    # ------------------------- 主循环 -------------------------
    def run(self):
        self.depth_viewer = DepthViewer(scale=6)
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            # start = time.time()
            while viewer.is_running():
                
                if self.should_reset():
                    self.reset_simulation()
                    viewer.sync()
                    continue

                step_start = time.time()
                if self.counter % self.control_decimation == 0:
                    self.build_observation()
                    # print(self.action)
                    raw_action_model = np.clip(
                        self.policy_inference(),
                        -self.clip_actions,
                        self.clip_actions,
                    ).astype(np.float32, copy=False)
                    self.raw_policy_action_model[:] = raw_action_model

                    raw_action_mj = raw_action_model[self.idx_mj_to_model]
                    filtered_action_mj = apply_action_filter(
                        raw_action_mj,
                        self.previous_action_mj,
                        self.action_filter_cfg,
                    )
                    self.action_model = filtered_action_mj[self.idx_model_to_mj].copy()
                    self.action_model[self.hip_indices_model] *= self.hip_scale_reduction
                    self.previous_action_model[:] = self.action_model
                    self.previous_action_mj[:] = self.action_model[self.idx_mj_to_model]
                    self.action_mj[:] = self.previous_action_mj

                    action_tensor = torch.from_numpy(self.raw_policy_action_model).to(self.device, dtype=torch.float32).view(1, 1, -1)
                    self.wm_action_history = torch.cat((self.wm_action_history[:, 1:], action_tensor), dim=1)
                    self.wm_action = self.wm_action_history.flatten(1)

                # points_flat = self.depth_points.reshape(-1, 3)
                # viewer.user_scn.ngeom = 0  # 清空
                # max_geom = viewer.user_scn.maxgeom
                # n = min(len(points_flat), max_geom)
                # for i in range(n):
                #     p = points_flat[i]
                #     mujoco.mjv_initGeom(
                #         viewer.user_scn.geoms[i],
                #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
                #         size=[0.008, 0.008, 0.008],   # 小一点，不然会糊成一片
                #         pos=p,
                #         mat=np.eye(3).flatten(),
                #         rgba=[1, 0, 0, 1],
                #     )
                # viewer.user_scn.ngeom = n

                self.counter += 1
                # ===== PD 控制 =====
                q_model = np.asarray(self.d.qpos[7:7 + self.num_actions], dtype=np.float32)[self.idx_model_to_mj]
                dq_model = np.asarray(self.d.qvel[6:6 + self.num_actions], dtype=np.float32)[self.idx_model_to_mj]
                action_scaled = self.action_model * self.action_scale
                tau_model = (
                    self.kps_model * (action_scaled + self.default_angles_model - q_model)
                    - self.kds_model * dq_model
                )
                wheel = self.wheel_indices_model
                tau_model[wheel] = (
                    self.wheel_kp_scale * action_scaled[wheel]
                    - self.wheel_kd_abs * dq_model[wheel]
                )
                tau_model = np.clip(tau_model, -self.torque_limits_model, self.torque_limits_model)
                self.d.ctrl[:] = tau_model[self.idx_mj_to_model]
                mujoco.mj_step(self.m, self.d)  # 5ms

                viewer.sync()
                now = time.time()
                if now - self.last_status_print_time > 0.1:
                    local_vel = quat_rotate_inverse(self.d.qpos[3:7], self.d.qvel[:3])
                    perception_status = ""
                    if self.debug_perception:
                        stats = self.last_depth_stats
                        perception_status = (
                            f" wm_norm={self.last_wm_norm: .2f}"
                            f" act_diff={self.last_act_diff_mean: .3f}/{self.last_act_diff_max: .3f}"
                            f" pre=[{stats['pre_raw_min']: .2f},{stats['pre_raw_max']: .2f}]"
                            f" depth=[{stats['raw_min']: .2f},{stats['raw_max']: .2f}]"
                            f" norm={stats['norm_mean']: .2f}/{stats['norm_std']: .2f}"
                            f" z={stats['base_z']: .2f} fz={stats['cam_forward_z']: .2f}"
                        )
                    print(
                        f"Cmd=[{self.cmd[0]: .2f},{self.cmd[1]: .2f},{self.cmd[2]: .2f}] "
                        f"Vel=[{local_vel[0]: .2f},{local_vel[1]: .2f}] "
                        f"wm_ms={self.last_wm_ms: .1f} policy_ms={self.last_policy_ms: .1f}"
                        f"{perception_status}",
                        end="\r",
                    )
                    self.last_status_print_time = now
                sleep_dt = self.m.opt.timestep - (time.time() - step_start)
                if sleep_dt > 0:
                    time.sleep(sleep_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name")
    parser.add_argument("scene_path", type=str, help="config file name")
    parser.add_argument("jit_load_path", type=str, help="config file name")
    parser.add_argument("--debug-perception", action="store_true", help="print WMP/depth action influence stats")
    parser.add_argument("--blind", action="store_true", help="run policy with wm_feature zeroed for comparison")
    args = parser.parse_args()
    
    sim = SimulationRunner(
        args.config_file,
        args.scene_path,
        args.jit_load_path,
        debug_perception=args.debug_perception,
        blind=args.blind,
    )
    sim.run()
