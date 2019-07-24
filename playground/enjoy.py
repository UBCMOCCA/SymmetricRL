"""
Helper script used for rendering the best learned policy for an existing experiment.

Usage:
```bash
python -m playground.enjoy with experiment_dir=runs/<EXPERIMENT_DIRECTORY>

# plot the joint positions over time
python -m playground.evaluate with experiment_dir=runs/<EXPERIMENT_DIRECTORY> plot=True
```
"""
import os
import time
from glob import glob
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

import torch

from common.envs_utils import make_env
from common.render_utils import StatsVisualizer
from common.sacred_utils import ex

import symmetry.sym_envs
from symmetry.env_utils import get_env_name_for_method


@ex.config
def config():
    net = None
    plot = False
    dump = False
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

    env = make_env(env_name, render=True)
    env.seed(1093)

    print("Env: {}".format(env_name))
    print("Model: {}".format(os.path.basename(model_path)))

    if args.dump:
        args.plot = False
        max_steps = 2000
        dump_dir = os.path.join(args.experiment_dir, "dump")
        image_sequence = []

        try:
            os.makedirs(dump_dir)
        except OSError:
            files = glob(os.path.join(dump_dir, "*.png"))
            for f in files:
                os.remove(f)

    else:
        max_steps = float("inf")

    if args.plot:
        num_steps = env.spec.max_episode_steps
        plotter = StatsVisualizer(100, num_steps)

    actor_critic = torch.load(model_path)

    states = torch.zeros(1, actor_critic.state_size)
    masks = torch.zeros(1, 1)

    obs = env.reset()

    ep_reward = 0
    prev_contact = False
    step = 0

    env_dt = getattr(env.unwrapped, "control_step", 1 / 60)
    start_time = time.time()

    while step < max_steps:
        step += 1

        if "Bullet" in args.env_name:
            env.unwrapped._p.resetDebugVisualizerCamera(
                3, 0, -5, env.unwrapped.robot.body_xyz
            )

        obs = torch.from_numpy(obs).float().unsqueeze(0)

        with torch.no_grad():
            value, action, _, states = actor_critic.act(
                obs, states, masks, deterministic=True
            )
        cpu_actions = action.squeeze().cpu().numpy()

        obs, reward, done, _ = env.step(cpu_actions)
        ep_reward += reward

        if args.plot:
            contact = env.unwrapped.robot.feet_contact[0] == 1
            strike = not prev_contact and contact
            plotter.update_plot(
                float(value),
                cpu_actions,
                ep_reward,
                done,
                strike,
                env.unwrapped.camera._fps,
            )
            prev_contact = contact

        if args.dump:
            image_sequence.append(env.unwrapped.camera.dump_rgb_array())

        if done:
            if not args.plot:
                print("Episode reward:", ep_reward)
            ep_reward = 0
            obs = env.reset()

        diff_time = env_dt * step - (time.time() - start_time)
        time.sleep(max(0, diff_time))

    if args.dump:
        import moviepy.editor as mp
        import datetime

        clip = mp.ImageSequenceClip(image_sequence, fps=60)
        now_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = os.path.join(dump_dir, "{}.mp4".format(now_string))
        clip.write_videofile(filename)
