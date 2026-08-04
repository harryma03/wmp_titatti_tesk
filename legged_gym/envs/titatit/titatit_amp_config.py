import glob
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

MOTION_FILES = glob.glob('datasets/mocap_motions/*')


class TitatitAMPCfg(LeggedRobotCfg):
    """Titatit wheeled-legged robot config for WMP training.

    Key differences from A1 quadruped:
    - 16 DOF: 12 leg joints + 4 wheel joints (indices 3,7,11,15)
    - Wheel joints use velocity/torque control, not PD position control
    - Action reindex between SDK order and Isaac order
    - Low-pass action filter
    - Hip scale reduction
    - Forward heightmap (525 points) instead of surrounding heightmap (187)
    """

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None

        # === Titatit specific dimensions ===
        num_dof = 16          # 12 leg + 4 wheel
        num_actions = 16
        action_dim = 16
        # Exact 16-DOF extension of the quadruped WMP contract.
        # This slice is passed to the world model and depth predictor:
        # [ang_vel(3), gravity(3), cmd(3), dof_pos(16), dof_vel(16)] = 41.
        # Previous actions are stored immediately after prop as a separate
        # action_dim block, just like the 12-DOF quadruped template.
        prop_dim = 41

        # WMP code uses privileged_dim as the OFFSET to proprioceptive data in the observation buffer.
        # obs buffer layout: [contact_flag(8), contact_force(12), domain_rand(38),
        #                     base_lin_vel(3),
        #                     ang_vel(3), gravity(3), cmd(3), dof_pos(16), dof_vel(16),
        #                     actions(16), heights(187)]
        #
        # domain_rand for 16 DOF: friction(1)+restitution(1)+mass(1)+com(3)+kp(16)+kd(16) = 38
        # contact: flag(8, thigh+calf×4legs) + force(12, 4feet×3) = 20
        # base_lin_vel is privileged-only, so the actor/WM offset is 58+3=61.
        privileged_dim = 38 + 8 + 12 + 3  # = 61 (offset to ang_vel)
        height_dim = 187      # SURROUNDING heightmap: 17 x 11 (used in compute_observations)
        forward_height_dim = 525  # forward heightmap: 21 x 25 (for depth predictor, accessed separately)

        # num_observations = full obs buffer size
        # = privileged_dim(61) + prop_dim(41) + action_dim(16) + height_dim(187) = 305
        num_observations = privileged_dim + prop_dim + action_dim + height_dim
        num_privileged_obs = num_observations

        reference_state_initialization = False
        amp_motion_files = MOTION_FILES

    class terrain:
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True

        # surrounding heightmap (187 points) - for critic
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        # forward heightmap (525 points) - for depth predictor
        measured_forward_points_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        measured_forward_points_y = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4,
                                     -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        # terrain types: [wave, rough slope, stairs up, stairs down, discrete, gap, pit, tilt, crawl, rough_flat]
        # 多地形混合训练
        #terrain_proportions = [0.10, 0.10, 0.15, 0.15, 0.10, 0.10, 0.10, 0.0, 0.0, 0.20]
        terrain_proportions = [0.05, 0.05, 0.25, 0.15, 0.0, 0.15, 0.15, 0.0, 0.0, 0.20]

        stair_height_base = 0.05
        stair_height_slope = 0.16
        tilt_width_base = 0.72
        tilt_width_slope = 0.08
        gap_size_scale = 0.30
        pit_depth_scale = 0.40
        crawl_height_base = 0.60
        crawl_height_slope = 0.08
        slope_treshold = 0.75

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.44]  # Titatit base height
        default_joint_angles = {
            'FL_hip_joint': 0.05,
            'RL_hip_joint': 0.05,
            'FR_hip_joint': -0.05,
            'RR_hip_joint': -0.05,

            'FL_thigh_joint': 1.0,
            'RL_thigh_joint': -0.8,
            'FR_thigh_joint': 1.0,
            'RR_thigh_joint': -0.8,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': 1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': 1.5,

            'FL_foot_joint': 0.0,   # wheel
            'RL_foot_joint': 0.0,   # wheel
            'FR_foot_joint': 0.0,   # wheel
            'RR_foot_joint': 0.0,   # wheel
        }

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 35.}  # all 16 DOFs get p_gains=25
        damping = {'joint': 0.625}  # all 16 DOFs get d_gains=0.625
        action_scale = 0.25
        decimation = 4
        hip_scale_reduction = 0.3

        # wheel override in wheeled_legged_robot.py:
        # torque = p_gains * wheel_kp * action_scaled - wheel_kd_scale * d_gains * vel
        wheel_kp = 0.4        # 25 * 0.4 = 10 effective kp
        wheel_kd_scale = 0.4  # 0.625 * 0.4 = 0.25 effective kd

        use_filter = True  # low-pass action filter

    class depth:
        use_camera = True
        use_warp_renderer = True
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.537, -0.004, 0.239]  # from URDF camera link inertial origin
        y_angle = [15, 25]   # positive = pitch DOWN in Isaac Gym from_euler_zyx[15.25]
        z_angle = [0, 0]
        x_angle = [0, 0]

        update_interval = 5
        min_non_crawl_tilt_camera_envs = 256


        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0.0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/titatit/urdf/wheeled_titatit_rl.urdf'
        foot_name = "foot"
        name = "titatit"
        penalize_contacts_on = ["thigh", "calf", "trunk"]
        terminate_after_contacts_on = ["trunk"]
        self_collisions = 0
        flip_visual_attachments = True

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.8, 1.6]
        randomize_restitution = True
        restitution_range = [0.0, 0.2]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = True
        com_x_pos_range = [-0.1, 0.1]
        com_y_pos_range = [-0.1, 0.1]
        com_z_pos_range = [-0.1, 0.1]

        push_robots = False
        push_interval_s = 15
        min_push_interval_s = 10
        max_push_vel_xy = 1.0

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_action_latency = True
        latency_range = [0.00, 0.005]

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            contact_force = 0.005
            com_pos = 20
            pd_gains = 5

        clip_observations = 100.
        clip_actions = 100.  # Titatit uses wider clip range
        base_height = 0.44   # Titatit base height

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.02
            contact_states = 0.05

    class rewards(LeggedRobotCfg.rewards):
        reward_curriculum = True
        reward_curriculum_term = [
            "feet_edge",
            "base_height_low",
            "collision",
            "cheat",
            "stuck",
            "front_wheel_impact",
            "front_touchdown_impact",
            "obstacle_front_clearance",
        ]
        reward_curriculum_schedule = [
            [4000, 10000, 0.1, 1.0],
            [0, 2000, 0.35, 1.0],
            [0, 2000, 0.45, 1.0],
            [0, 2000, 0.25, 1.0],
            [0, 2000, 0.20, 1.0],
            [0, 2500, 0.35, 1.0],
            [0, 3000, 0.00, 1.0],
            [1000, 4000, 0.10, 1.0],
        ]

        soft_dof_pos_limit = 0.9                    # 关节软限位比例；越小越早惩罚接近关节极限
        base_height_target = 0.44                   # 基础机身高度参考值，供通用高度类奖励使用
        foot_height_target = 0.12                   # 摆腿/抬脚高度参考值，供部分脚高奖励使用
        tracking_sigma = 0.10                       # 速度跟踪奖励宽度；越小越严格要求贴近目标速度
        lin_vel_clip = 0.1                          # 线速度跟踪容忍带；目标速度附近 +/- 该值不继续惩罚
        gait_contact_force_threshold = 1.0          # 判断轮/脚是否接触地面的竖直力阈值
        terrain_progress_x0 = 2                     # 前方高度图中用于检测障碍起点的 x 索引
        terrain_progress_x1 = 12                    # 前方高度图中用于检测障碍终点的 x 索引
        terrain_progress_y_radius = 4               # 前方检测区域的横向半宽索引，越大看得越宽
        terrain_progress_trigger_height = 0.04      # 前方高度变化超过该值时，认为 terrain_progress 激活

        terrain_progress_impact_width = 0.50        # 前轮撞击削弱 progress 的宽度；越小撞击越容易压低前进奖励
        terrain_progress_clearance_width = 0.60     # 清障 deficit/clearance 相关归一化宽度
        terrain_corridor_deadband = 0.35            # 走廊中心允许横向偏差；超过后 progress 权重下降
        terrain_corridor_heading_width = 0.7        # 航向角偏差宽度；越小越要求朝正前方过障
        lateral_deviation_deadband = 0.25           # lateral_deviation 惩罚死区；小偏移不扣分
        obstacle_tracking_lin_vel_x = 0.40          # 非 gap 障碍附近的 x 速度跟踪上限/目标
        gap_tracking_lin_vel_x = 0.40               # gap 附近的 x 速度跟踪上限/目标
        terrain_progress_max_vel = 1.0              # 实际生效的前进奖励速度饱和值
        front_wheel_impact_force_threshold = 8.0    # 前轮水平冲击力超过该值才开始计入撞击
        front_wheel_impact_force_scale = 20.0       # 水平冲击力归一化尺度；越大撞击分数增长越慢
        front_wheel_impact_vertical_ratio = 0.25    # 水平力/竖直力比例阈值，用于区分滚动支撑和撞边
        front_wheel_impact_ratio_scale = 0.6        # 水平/竖直力比例的归一化尺度
        front_wheel_impact_speed_threshold = 0.10   # 基座前进速度超过该值时，前轮撞击惩罚开始生效
        front_wheel_impact_speed_scale = 0.20       # 前进速度归一化尺度；越小速度门控越快到满
        front_wheel_impact_max_penalty = 2.5        # 单步前轮撞击原始分数上限，避免极端接触爆炸
        front_clearance_light_contact_force = 6.0   # 判断前轮是否轻载/卸载的竖直力阈值，用于清障摆腿奖励
        front_clearance_horizontal_force = 8.0      # 判断前轮是否有明显水平顶撞的力阈值
        early_front_clearance_x0 = 5                # 提前清障检测窗口起点 x 索引
        early_front_clearance_x1 = 18               # 提前清障检测窗口终点 x 索引
        early_front_clearance_trigger_height = 0.03 # 提前清障触发高度变化阈值
        obstacle_front_air_trigger_height = 0.055   # 前轮腾空/air-time 奖励触发的最小高度变化
        obstacle_front_air_preimpact_force = 8.0    # 腾空序列开始前若水平力超过该值，认为已撞上障碍
        obstacle_front_air_preimpact_score = 0.05   # 腾空序列开始前若撞击分数超过该值，阻断该次奖励
        obstacle_front_clearance_margin = 0.06      # 前轮清障目标在障碍高度上额外加的安全余量
        obstacle_front_clearance_min_target = 0.12  # 非 gap 前轮清障目标高度下限
        obstacle_front_clearance_max_target = 0.24  # 非 gap 前轮清障目标高度上限
        obstacle_front_clearance_far_distance = 1.6 # 远距离开始逐渐要求前轮 clearance 的距离
        obstacle_front_clearance_deadline_distance = 0.75 # 到该距离附近仍未清障会更强惩罚
        stair_front_clearance_deadline_distance = 1.05    # 楼梯专用前轮 clearance 截止距离
        stair_front_clearance_close_distance = 0.55       # 楼梯很近距离，用于 deadline 权重拉满
        pit_front_wheel_impact_weight = 0.45       # pit 地形前轮撞击惩罚权重系数，低于普通障碍
        pit_progress_impact_weight = 0.25          # pit 地形中撞击削弱 progress 的系数，低于普通障碍
        pit_stuck_multiplier = 1.6                 # pit 地形卡住惩罚倍率
        obstacle_front_swing_start_distance = 1.65 # 前轮摆动相位窗口开始距离；更远开始准备抬前轮
        obstacle_front_swing_peak_distance = 1.18  # 前轮摆动奖励峰值距离；此处最鼓励清障姿态
        obstacle_front_swing_end_distance = 0.60   # 前轮摆动相位窗口结束距离
        obstacle_front_swing_clearance_margin = 0.03 # 摆动清障目标在障碍高度上的额外余量
        obstacle_front_swing_min_lift = 0.015      # 认为前轮开始有效抬起的最小高度
        obstacle_front_swing_min_clearance = 0.06  # 摆动清障目标高度下限
        obstacle_front_swing_max_clearance = 0.18  # 摆动清障目标高度上限，避免鼓励抬太高
        obstacle_front_swing_impact_threshold = 14.0 # 摆动过程中水平力低于该值才算 clean swing
        obstacle_front_air_min_time = 0.08         # 前轮腾空至少持续该时间，落地事件才给 air-time 奖励
        obstacle_front_air_stair_target_time = 0.16 # stair 前轮腾空目标时间
        obstacle_front_air_gap_target_time = 0.22  # gap 前轮腾空目标时间
        obstacle_front_air_pit_target_time = 0.22  # pit 前轮腾空目标时间
        obstacle_front_air_max_time = 0.36         # 非 gap 前轮腾空最长有效时间，超过后奖励衰减
        obstacle_front_air_gap_max_time = 0.45     # gap 前轮腾空最长有效时间，允许比 stair 更久
        obstacle_front_air_landing_force = 10.0    # 落地水平力低于该值才算干净落地
        obstacle_front_air_lateral_speed = 0.35    # 落地/腾空期间前轮侧向速度阈值，越小越限制横摆
        obstacle_front_air_down_speed = 0.65       # 前轮向下落地速度阈值，越小越要求轻落地
        obstacle_front_air_progress_vel = 0.35     # air-time 奖励中的前进速度门槛/归一化值
        obstacle_front_air_stair_weight = 1.0      # stair 地形 air-time 奖励权重
        obstacle_front_air_gap_weight = 0.9        # gap 地形 air-time 奖励权重
        obstacle_front_air_pit_weight = 0.90       # pit 地形 air-time 奖励权重
        obstacle_front_edge_prepare_distance = 1.50 # 接近边缘前开始准备 clearance 的距离
        obstacle_front_edge_deadline_distance = 0.95 # 到该距离仍未清边，edge fail 权重开始变强
        obstacle_front_edge_close_distance = 0.45  # 非常接近边缘的距离，edge fail 权重接近满
        obstacle_front_edge_clean_force = 6.0      # 清边时允许的最大水平力，越小越要求不顶边
        obstacle_front_edge_progress_vel = 0.35    # edge clearance 奖励中的前进速度归一化值
        obstacle_front_edge_stuck_speed = 0.10     # 低于该前进速度且在边缘附近，认为有卡住趋势
        base_height_low_target = 0.40              # 机身过低惩罚目标高度；低于该值开始扣
        base_height_low_margin = 0.10              # 机身过低惩罚的归一化宽度

        class scales:
            
            tracking_lin_vel = 1.6
            tracking_ang_vel = 0.6
            torques = -0.0001
            dof_acc = -3.5e-7
            base_height_low = -0.35
            collision = -1.0
            feet_stumble = -0.6
            feet_edge = -1.0
            front_wheel_impact = -2.0
            front_touchdown_impact = -2.0
            obstacle_front_lift = 1.0
            front_pair_air = -0.10
            front_swing_clearance = 0.25
            action_rate = -0.04
            dof_error = -0.05
            hip_deviation = -0.25
            rear_leg_deviation = -0.22

            lin_vel_z = -1.0
            ang_vel_xy = -0.12
            orientation = -1.6
            cheat = -1.2            # penalize going around obstacles
            stuck = -0.45

            terrain_progress = 2.2      # forward progress through terrain, gated by corridor and impact
            lateral_deviation = -0.2    # weak guard; main corridor term is terrain_progress
            obstacle_front_clearance = -0.20
            stair_front_clearance_deadline = 0.0
            feet_air_time = 1.5
            gap_front_pair_clearance = 0.25

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_lin_vel_forward_x_curriculum = 0.8
        max_lin_vel_backward_x_curriculum = 0.0
        max_lin_vel_y_curriculum = 0.0
        max_ang_vel_yaw_curriculum = 1.0

        max_flat_lin_vel_forward_x_curriculum = 0.65
        max_flat_lin_vel_backward_x_curriculum = 0.0
        max_flat_lin_vel_y_curriculum = 0.0
        max_flat_ang_vel_yaw_curriculum = 1.4

        num_commands = 4
        resampling_time = 10.
        heading_command = False  # forward-crossing task first

        class ranges:
            lin_vel_x = [0.0, 1.5]
            lin_vel_y = [-0.15, 0.15]
            ang_vel_yaw = [-1.2, 1.2]
            heading = [-3.14 / 4, 3.14 / 4]

            flat_lin_vel_x = [-1.0, 2.0]
            flat_lin_vel_y = [-0.5, 0.5]
            flat_ang_vel_yaw = [-1.4, 1.4]
            flat_heading = [-3.14 / 4, 3.14 / 4]


class TitatitAMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'WMPRunner'

    class policy:
        init_noise_std = 0.8
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3  # history latent + command
        wm_latent_dim = 32
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        vel_predict_coef = 1.0
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'WMP_titatit_v23_front_air_gait'
        experiment_name = 'titatit_amp_example'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCriticWMP'
        max_iterations = 10000
        save_interval = 100

        amp_reward_coef = 0.5 * 0.02
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        # 16 actions: leg joints get tighter std, wheel joints get wider
        min_normalized_std = [0.05, 0.02, 0.05, 0.1] * 4

    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100
