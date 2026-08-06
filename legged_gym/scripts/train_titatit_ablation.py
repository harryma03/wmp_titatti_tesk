import os
import inspect

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


VARIANTS = {
    "a1_style": {
        "wheel_realization": False,
        "event_rewards": False,
    },
    "wheel_only": {
        "wheel_realization": True,
        "event_rewards": False,
    },
    "event_only": {
        "wheel_realization": False,
        "event_rewards": True,
    },
    "full": {
        "wheel_realization": True,
        "event_rewards": True,
    },
}

REPRESENTATIONS = ("rssm", "drd")

EVENT_REWARD_TERMS = (
    "feet_edge",
    "front_wheel_impact",
    "front_touchdown_impact",
    "obstacle_front_lift",
    "front_pair_air",
    "front_swing_clearance",
    "cheat",
    "stuck",
    "terrain_progress",
    "lateral_deviation",
    "obstacle_front_clearance",
    "feet_air_time",
    "gap_front_pair_clearance",
    "stair_front_clearance_deadline",
)


def configure_ablation(env_cfg, variant_name):
    if variant_name not in VARIANTS:
        raise ValueError(
            f"Unknown WL_ABLATION={variant_name!r}; "
            f"choose one of {tuple(VARIANTS)}"
        )

    variant = VARIANTS[variant_name]
    use_wheel = variant["wheel_realization"]
    use_events = variant["event_rewards"]

    env_cfg.control.use_wheel_realization = use_wheel
    env_cfg.control.zero_wheel_position_obs = use_wheel
    env_cfg.rewards.use_event_rewards = use_events

    if not use_events:
        for name in EVENT_REWARD_TERMS:
            if not hasattr(env_cfg.rewards.scales, name):
                raise AttributeError(
                    f"Missing reward scale: {name}"
                )
            setattr(env_cfg.rewards.scales, name, 0.0)

    return variant


def validate_representation(representation_name, variant_name):
    if representation_name not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown WL_REPRESENTATION={representation_name!r}; "
            f"choose one of {REPRESENTATIONS}"
        )

    if representation_name == "drd" and variant_name != "full":
        raise ValueError(
            "WL_REPRESENTATION=drd must be used with "
            "WL_ABLATION=full. The DRD experiment changes only "
            "the representation backend."
        )


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(
        name=args.task
    )

    variant_name = os.environ.get(
        "WL_ABLATION", "full"
    ).strip().lower()

    representation_name = os.environ.get(
        "WL_REPRESENTATION", "rssm"
    ).strip().lower()

    seed = int(
        os.environ.get("WL_SEED", "1")
    )

    max_iterations = int(
        os.environ.get("WL_MAX_ITERS", "10000")
    )

    num_envs_override = os.environ.get(
        "WL_NUM_ENVS"
    )

    validate_representation(
        representation_name,
        variant_name,
    )

    variant = configure_ablation(
        env_cfg,
        variant_name,
    )

    os.environ["WL_REPRESENTATION"] = (
        representation_name
    )

    env_cfg.seed = seed

    if num_envs_override is not None:
        env_cfg.env.num_envs = int(
            num_envs_override
        )

    train_cfg.seed = seed

    train_cfg.runner.run_name = (
        f"WMP_rep_{representation_name}_"
        f"{variant_name}_seed{seed}"
    )

    train_cfg.runner.load_run = ""
    train_cfg.runner.checkpoint = -1
    train_cfg.runner.max_iterations = (
        max_iterations
    )
    train_cfg.runner.save_interval = 1000
    train_cfg.runner.resume = False

    train_cfg.algorithm.learning_rate = 5.0e-4
    train_cfg.algorithm.entropy_coef = 0.004
    train_cfg.algorithm.num_learning_epochs = 5

    print("=" * 72)
    print(f"representation      : {representation_name}")
    print(f"variant             : {variant_name}")
    print(f"seed                : {seed}")
    print(f"wheel realization   : {variant['wheel_realization']}")
    print(f"event rewards       : {variant['event_rewards']}")
    print(f"max iterations      : {max_iterations}")
    print(f"num envs            : {env_cfg.env.num_envs}")
    print(f"run name            : {train_cfg.runner.run_name}")
    print("=" * 72)

    env, env_cfg = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )

    ppo_runner, train_cfg = (
        task_registry.make_wmp_runner(
            env=env,
            name=args.task,
            args=args,
            train_cfg=train_cfg,
        )
    )

    ppo_runner.learn(
        num_learning_iterations=(
            train_cfg.runner.max_iterations
        ),
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    args.rl_device = args.sim_device
    train(args)