# # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: BSD-3-Clause
# # 
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# # list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# # this list of conditions and the following disclaimer in the documentation
# # and/or other materials provided with the distribution.
# #
# # 3. Neither the name of the copyright holder nor the names of its
# # contributors may be used to endorse or promote products derived from
# # this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #
# # Copyright (c) 2021 ETH Zurich, Nikita Rudin

# # This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# # All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

# import os
# import copy
# import torch
# import numpy as np
# import random
# from isaacgym import gymapi
# from isaacgym import gymutil

# from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

# def class_to_dict(obj) -> dict:
#     if not  hasattr(obj,"__dict__"):
#         return obj
#     result = {}
#     for key in dir(obj):
#         if key.startswith("_"):
#             continue
#         element = []
#         val = getattr(obj, key)
#         if isinstance(val, list):
#             for item in val:
#                 element.append(class_to_dict(item))
#         else:
#             element = class_to_dict(val)
#         result[key] = element
#     return result

# def update_class_from_dict(obj, dict):
#     for key, val in dict.items():
#         attr = getattr(obj, key, None)
#         if isinstance(attr, type):
#             update_class_from_dict(attr, val)
#         else:
#             setattr(obj, key, val)
#     return

# def set_seed(seed):
#     if seed == -1:
#         seed = np.random.randint(0, 10000)
#     print("Setting seed: {}".format(seed))
    
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def parse_sim_params(args, cfg):
#     # code from Isaac Gym Preview 2
#     # initialize sim params
#     sim_params = gymapi.SimParams()

#     # set some values from args
#     if args.physics_engine == gymapi.SIM_FLEX:
#         if args.device != "cpu":
#             print("WARNING: Using Flex with GPU instead of PHYSX!")
#     elif args.physics_engine == gymapi.SIM_PHYSX:
#         sim_params.physx.use_gpu = args.use_gpu
#         sim_params.physx.num_subscenes = args.subscenes
#     sim_params.use_gpu_pipeline = args.use_gpu_pipeline

#     # if sim options are provided in cfg, parse them and update/override above:
#     if "sim" in cfg:
#         gymutil.parse_sim_config(cfg["sim"], sim_params)

#     # Override num_threads if passed on the command line
#     if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
#         sim_params.physx.num_threads = args.num_threads

#     return sim_params

# def get_load_path(root, load_run=-1, checkpoint=-1):
#     try:
#         runs = os.listdir(root)
#         #TODO sort by date to handle change of month
#         runs.sort()
#         if 'exported' in runs: runs.remove('exported')
#         last_run = os.path.join(root, runs[-1])
#     except:
#         raise ValueError("No runs in this directory: " + root)
#     if load_run==-1:
#         load_run = last_run
#     else:
#         load_run = os.path.join(root, load_run)

#     if checkpoint==-1:
#         models = [file for file in os.listdir(load_run) if 'model' in file]
#         models.sort(key=lambda m: '{0:0>15}'.format(m))
#         model = models[-1]
#     else:
#         model = "model_{}.pt".format(checkpoint) 

#     load_path = os.path.join(load_run, model)
#     return load_path

# def update_cfg_from_args(env_cfg, cfg_train, args):
#     # seed
#     if env_cfg is not None:
#         # num envs
#         if args.num_envs is not None:
#             env_cfg.env.num_envs = args.num_envs
#     if cfg_train is not None:
#         if args.seed is not None:
#             cfg_train.seed = args.seed
#         # alg runner parameters
#         if args.max_iterations is not None:
#             cfg_train.runner.max_iterations = args.max_iterations
#         if args.resume:
#             cfg_train.runner.resume = args.resume
#         if args.experiment_name is not None:
#             cfg_train.runner.experiment_name = args.experiment_name
#         if args.run_name is not None:
#             cfg_train.runner.run_name = args.run_name
#         if args.load_run is not None:
#             cfg_train.runner.load_run = args.load_run
#         if args.checkpoint is not None:
#             cfg_train.runner.checkpoint = args.checkpoint

#     return env_cfg, cfg_train

# def get_args():
#     custom_parameters = [
#         {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
#         {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
#         {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
#         {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
#         {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
#         {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
#         {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
#         {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
#         {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
#         {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
#         {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
#         {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
#         {"name": "--terrain", "type": str, "default": "climb",
#          "help": 'Only for play'},
#         {"name": "--terrain_level", "type": int,
#          "help": 'Only for play. Fix all envs to this terrain row/level, e.g. 1 or 2.'},
#         {"name": "--play_warmup_steps", "type": int, "default": 0,
#          "help": 'Only for play. Warm up depth/world-model state before the policy starts.'},
#         {"name": "--play_warmup_mode", "type": str, "default": "latent",
#          "help": 'Only for play. latent: warm up world-model state without stepping physics; sim: run zero-action physics steps.'},
#         {"name": "--play_debug_metrics", "action": "store_true", "default": False,
#          "help": 'Only for play. Print obstacle, clearance, impact, progress, and posture metrics during rollout.'},
#         {"name": "--wm_device", "type": str, "default": "None", "help": 'World model device. Overrides config file in dreamer/config.yaml if provided'},

#     ]
#     # parse arguments
#     args = gymutil.parse_arguments(
#         description="RL Policy",
#         custom_parameters=custom_parameters)

#     # name allignment
#     args.sim_device_id = args.compute_device_id
#     args.sim_device = args.sim_device_type
#     if args.sim_device=='cuda':
#         args.sim_device += f":{args.sim_device_id}"
#     return args

# def export_policy_as_jit(actor_critic, path):
#     if hasattr(actor_critic, 'memory_a'):
#         # assumes LSTM: TODO add GRU
#         exporter = PolicyExporterLSTM(actor_critic)
#         exporter.export(path)
#     else: 
#         os.makedirs(path, exist_ok=True)
#         path = os.path.join(path, 'policy_1.pt')
#         model = copy.deepcopy(actor_critic.actor).to('cpu')
#         traced_script_module = torch.jit.script(model)
#         traced_script_module.save(path)


# class PolicyExporterWMPOnnx(torch.nn.Module):
#     def __init__(self, actor_critic):
#         super().__init__()
#         self.history_encoder = copy.deepcopy(actor_critic.history_encoder)
#         self.wm_feature_encoder = copy.deepcopy(actor_critic.wm_feature_encoder)
#         self.actor = copy.deepcopy(actor_critic.actor)
#         self.privileged_dim = actor_critic.privileged_dim

#     def forward(self, obs, history, wm_feature):
#         latent = self.history_encoder(history)
#         command = obs[:, self.privileged_dim + 6:self.privileged_dim + 9]
#         wm_latent = self.wm_feature_encoder(wm_feature)
#         actor_input = torch.cat((latent, command, wm_latent), dim=-1)
#         return self.actor(actor_input)


# def export_policy_as_onnx(actor_critic, path, obs_dim, history_dim, wm_feature_dim, filename='policy_wmp.onnx'):
#     if not hasattr(actor_critic, 'history_encoder') or not hasattr(actor_critic, 'wm_feature_encoder'):
#         raise ValueError("ONNX export here expects an ActorCriticWMP policy")

#     os.makedirs(path, exist_ok=True)
#     output_path = os.path.join(path, filename)
#     exporter = PolicyExporterWMPOnnx(actor_critic).to('cpu').eval()

#     dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
#     dummy_history = torch.zeros(1, history_dim, dtype=torch.float32)
#     dummy_wm_feature = torch.zeros(1, wm_feature_dim, dtype=torch.float32)

#     torch.onnx.export(
#         exporter,
#         (dummy_obs, dummy_history, dummy_wm_feature),
#         output_path,
#         export_params=True,
#         opset_version=11,
#         do_constant_folding=True,
#         input_names=['obs', 'history', 'wm_feature'],
#         output_names=['actions'],
#         dynamic_axes={
#             'obs': {0: 'batch_size'},
#             'history': {0: 'batch_size'},
#             'wm_feature': {0: 'batch_size'},
#             'actions': {0: 'batch_size'},
#         },
#     )
#     return output_path


# class PolicyExporterLSTM(torch.nn.Module):
#     def __init__(self, actor_critic):
#         super().__init__()
#         self.actor = copy.deepcopy(actor_critic.actor)
#         self.is_recurrent = actor_critic.is_recurrent
#         self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
#         self.memory.cpu()
#         self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
#         self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

#     def forward(self, x):
#         out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
#         self.hidden_state[:] = h
#         self.cell_state[:] = c
#         return self.actor(out.squeeze(0))

#     @torch.jit.export
#     def reset_memory(self):
#         self.hidden_state[:] = 0.
#         self.cell_state[:] = 0.
 
#     def export(self, path):
#         os.makedirs(path, exist_ok=True)
#         path = os.path.join(path, 'policy_lstm_1.pt')
#         self.to('cpu')
#         traced_script_module = torch.jit.script(self)
#         traced_script_module.save(path)

    
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
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from utils.export_inference_model import *

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--terrain", "type": str, "default": "climb",
         "help": 'Only for play'},
        {"name": "--terrain_level", "type": int,
         "help": 'Only for play. Fix all envs to this terrain row/level, e.g. 1 or 6.'},
        {"name": "--wm_device", "type": str, "default": "None", "help": 'World model device. Overrides config file in dreamer/config.yaml if provided'},

    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    # args.sim_device_id = args.compute_device_id
    # args.sim_device = args.sim_device_type
    # if args.sim_device=='cuda':
    #     args.sim_device += f":{args.sim_device_id}"
    args.sim_device = args.rl_device

    return args


def export_3m_as_jit(actor_critic,world_model, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        hist_enc = copy.deepcopy(actor_critic.history_encoder).cpu()
        torch.jit.script(hist_enc).save(os.path.join(path, "hist_enc.pt"))

        wm_feat_enc = copy.deepcopy(actor_critic.wm_feature_encoder).cpu()
        torch.jit.script(wm_feat_enc).save(os.path.join(path, "wm_enc.pt"))

        actor = copy.deepcopy(actor_critic.actor).cpu()
        torch.jit.script(actor).save(os.path.join(path, "actor.pt"))


def _first_linear_in_features(module):
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Linear):
            return int(submodule.in_features)
    raise ValueError(f"No Linear layer found in {module.__class__.__name__}")


def _world_model_export_dims(world_model):
    encoder = world_model.encoder
    prop_dim = None
    image_shape = (64, 64, 1)

    mlp_shapes = getattr(encoder, "mlp_shapes", None)
    if mlp_shapes:
        prop_dim = int(sum(int(np.prod(shape)) for shape in mlp_shapes.values()))

    cnn_shapes = getattr(encoder, "cnn_shapes", None)
    if cnn_shapes:
        image_shape = tuple(int(v) for v in next(iter(cnn_shapes.values())))

    if prop_dim is None and hasattr(encoder, "_mlp"):
        prop_dim = _first_linear_in_features(encoder._mlp)
    if prop_dim is None:
        raise ValueError("Could not infer world model prop_dim for export")

    return prop_dim, image_shape


def export_2m_as_jit(actor_critic, world_model, path):
    os.makedirs(path, exist_ok=True)
    # wm_copy = copy.deepcopy(world_model)
    actor_copy = copy.deepcopy(actor_critic)

    # ==========================
    # 2️⃣ 导出 World Model（trace）
    # ==========================
    wm_model = WorldModelInference(world_model).cpu().eval()

    prop_dim, image_shape = _world_model_export_dims(world_model)
    dummy_prop = torch.zeros(1, prop_dim)
    dummy_img = torch.zeros((1,) + image_shape)
    dummy_action = torch.zeros(1, world_model.dynamics._num_actions)
    dummy_is_first = torch.zeros(1)

    # world_model may currently live on CUDA (play_titatit moves it to the
    # environment device). The export wrapper and all dummy inputs are on CPU,
    # so move the recurrent dummy state to CPU as well.
    dummy_state = {
        key: value.detach().cpu()
        for key, value in world_model.dynamics.initial(1).items()
    }

    traced_wm = torch.jit.trace(
        wm_model,
        (
            dummy_prop,
            dummy_img,
            dummy_state["logit"],
            dummy_state["stoch"],
            dummy_state["deter"],
            dummy_action,
            dummy_is_first,
        ),
    )
    traced_wm = torch.jit.freeze(traced_wm)  # 固化参数，去掉无用属性
    # traced_wm = torch.jit.optimize_for_inference(traced_wm)
    quantized_model = torch.quantization.quantize_dynamic(
        traced_wm, 
        {torch.nn.Linear},  # 对线性层进行量化
        dtype=torch.qint8  # 使用 8 位整数量化
    )
    quantized_model.save(os.path.join(path, "world_model.pt"))
    # traced_wm.save(os.path.join(path, "world_model.pt"))
    print("World model exported.")

    # ==========================
    # 3️⃣ 导出 Policy（script）
    # ========================== 
    policy_model = PolicyInference(actor_copy).cpu().eval()

    scripted_policy = torch.jit.script(policy_model)
    scripted_policy.save(os.path.join(path, "policy.pt"))

    print("Policy exported.")



def export_2m_as_onnx(actor_critic, world_model, path):
    os.makedirs(path, exist_ok=True)

    actor_copy = copy.deepcopy(actor_critic)

    # ==========================
    # 1️⃣ World Model → ONNX
    # ==========================
    wm_model = WorldModelInference(world_model).cpu().eval()

    prop_dim, image_shape = _world_model_export_dims(world_model)
    dummy_prop = torch.zeros(1, prop_dim)
    dummy_img = torch.zeros((1,) + image_shape)
    dummy_action = torch.zeros(1, world_model.dynamics._num_actions)
    dummy_is_first = torch.zeros(1)

    # Keep every trace input on the same device as wm_model. Without this,
    # exporting from play_titatit mixes CUDA recurrent state with CPU inputs.
    dummy_state = {
        key: value.detach().cpu()
        for key, value in world_model.dynamics.initial(1).items()
    }

    dummy_inputs = (
        dummy_prop,
        dummy_img,
        dummy_state["logit"],
        dummy_state["stoch"],
        dummy_state["deter"],
        dummy_action,
        dummy_is_first,
    )

    input_names = [
        "prop",
        "img",
        "logit",
        "stoch",
        "deter",
        "action",
        "is_first",
    ]

    output_names = [
        "out_logit",
        "out_stoch",
        "out_deter",
        "wm_feature",
    ]

    torch.onnx.export(
        wm_model,
        dummy_inputs,
        os.path.join(path, "world_model.onnx"),
        input_names=input_names,
        output_names=output_names,
        opset_version=15,
        do_constant_folding=False,
        dynamic_axes={
            "prop": {0: "batch"},
            "img": {0: "batch"},
            "logit": {0: "batch"},
            "stoch": {0: "batch"},
            "deter": {0: "batch"},
            "action": {0: "batch"},
            "is_first": {0: "batch"},
        },
    )

    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=os.path.join(path, "world_model.onnx"),
        model_output=os.path.join(path, "world_model_int8.onnx"),
        weight_type=QuantType.QInt8,

        # 只量化 Linear， 量化卷积层会报错
        op_types_to_quantize=["MatMul", "Gemm"],

        per_channel=True
    )
    print("World model ONNX exported.")

    # ==========================
    # 2️⃣ Policy → ONNX
    # ==========================
    policy_model = PolicyInference(actor_copy).cpu().eval()

    history_dim = _first_linear_in_features(actor_copy.history_encoder)
    wm_feature_dim = _first_linear_in_features(actor_copy.wm_feature_encoder)
    dummy_command = torch.zeros(1, 3)
    dummy_history = torch.zeros(1, history_dim)
    dummy_wm_feature = torch.zeros(1, wm_feature_dim)
    print(
        "2M ONNX export dims: "
        f"prop={prop_dim}, image={image_shape}, "
        f"action={world_model.dynamics._num_actions}, "
        f"history={history_dim}, wm_feature={wm_feature_dim}"
    )
    torch.onnx.export(
        policy_model,
        (dummy_command, dummy_history, dummy_wm_feature),
        os.path.join(path, "policy.onnx"),
        input_names=["command", "history", "wm_feature"],
        output_names=["action"],
        opset_version=15,
        do_constant_folding=False,
        dynamic_axes={
            "command": {0: "batch"},
            "history": {0: "batch"},
            "wm_feature": {0: "batch"},
        },
    )

    print("Policy ONNX exported.")


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(model_path)


class PolicyExporterWMPOnnx(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.history_encoder = copy.deepcopy(actor_critic.history_encoder)
        self.wm_feature_encoder = copy.deepcopy(actor_critic.wm_feature_encoder)
        self.actor = copy.deepcopy(actor_critic.actor)
        self.privileged_dim = actor_critic.privileged_dim

    def forward(self, obs, history, wm_feature):
        latent = self.history_encoder(history)
        command = obs[:, self.privileged_dim + 6:self.privileged_dim + 9]
        wm_latent = self.wm_feature_encoder(wm_feature)
        actor_input = torch.cat((latent, command, wm_latent), dim=-1)
        return self.actor(actor_input)


def export_policy_as_onnx(actor_critic, path, obs_dim, history_dim, wm_feature_dim, filename='policy_wmp.onnx'):
    if not hasattr(actor_critic, 'history_encoder') or not hasattr(actor_critic, 'wm_feature_encoder'):
        raise ValueError("ONNX export here expects an ActorCriticWMP policy")

    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, filename)
    exporter = PolicyExporterWMPOnnx(actor_critic).to('cpu').eval()

    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    dummy_history = torch.zeros(1, history_dim, dtype=torch.float32)
    dummy_wm_feature = torch.zeros(1, wm_feature_dim, dtype=torch.float32)

    torch.onnx.export(
        exporter,
        (dummy_obs, dummy_history, dummy_wm_feature),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['obs', 'history', 'wm_feature'],
        output_names=['actions'],
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'history': {0: 'batch_size'},
            'wm_feature': {0: 'batch_size'},
            'actions': {0: 'batch_size'},
        },
    )
    return output_path


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
