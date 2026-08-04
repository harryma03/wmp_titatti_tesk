import numpy as np
import os
from datetime import datetime

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


#python legged_gym/scripts/train_titatit.py --task=titatit_amp --headless
def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # The 41-D world-model prop changes the network architecture, so this run
    # must start from scratch in a new directory. Old 57/60-D checkpoints are
    # intentionally not resumed.
    train_cfg.runner.run_name = 'WMP_titatit_0717'
    train_cfg.runner.load_run = 'WMP_titatit_quad_template_41'
    train_cfg.runner.checkpoint = -1
    train_cfg.runner.max_iterations = 50000
    train_cfg.runner.save_interval = 1000
    train_cfg.runner.resume = False

    train_cfg.algorithm.learning_rate = 5.e-4
    train_cfg.algorithm.entropy_coef = 0.004
    train_cfg.algorithm.num_learning_epochs = 5

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_wmp_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    args.rl_device = args.sim_device
    train(args)
