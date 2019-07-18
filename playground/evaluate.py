import os
import time
import json
from glob import glob
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

import torch

from common.envs_utils import make_env
from common.render_utils import StatsVisualizer
from common.sacred_utils import ex
from symmetry.metric_utils import MetricsEnv

import symmetry.sym_envs
from symmetry.env_utils import get_env_name_for_method


@ex.config
def config():
    net = None
    render = False
    max_steps = 5000
    env_name = ""
    experiment_dir = "."
    assert experiment_dir != "."
    ex.add_config(os.path.join(experiment_dir, "configs.json"))  # loads saved configs


@ex.automain
def main(_config):
    args = SimpleNamespace(**_config)
    assert args.env_name != ""

    env_name = args.env_name
    env_name = get_env_name_for_method(args.env_name, args.mirror_method)

    model_path = args.net or os.path.join(
        args.experiment_dir, "models", "{}_best.pt".format(env_name)
    )

    env = make_env(env_name, render=args.render)

    # TODO: assert Walker3D or Bullet
    env = MetricsEnv(env)
    env.seed(1093)

    print("Env: {}".format(env_name))
    print("Model: {}".format(os.path.basename(model_path)))

    actor_critic = torch.load(model_path)

    ep_reward = 0

    states = torch.zeros(1, actor_critic.state_size)
    masks = torch.zeros(1, 1)
    obs = env.reset()

    metrics = {}

    for step in range(args.max_steps):
        obs = torch.from_numpy(obs).float().unsqueeze(0)

        with torch.no_grad():
            value, action, _, states = actor_critic.act(
                obs, states, masks, deterministic=True
            )
        cpu_actions = action.squeeze().cpu().numpy()

        obs, reward, done, info = env.step(cpu_actions)

        if "Bullet" in env_name:
            env.unwrapped._p.resetDebugVisualizerCamera(
                3, 0, -5, env.unwrapped.robot.body_xyz
            )

        if "metrics" in info:
            if args.render:
                print(info["metrics"])
                time.sleep(1)
            for k, v in info["metrics"].items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

        if done:
            ep_reward = 0
            obs = env.reset()

    for k, v in metrics.items():
        metrics[k] = torch.FloatTensor(v).median().item()
        print("%s SI: %5.1f" % (k, metrics[k]))

    with open(os.path.join(args.experiment_dir, "evaluate.json"), "w") as eval_file:
        json.dump(metrics, eval_file)
