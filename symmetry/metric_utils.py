import torch as th
import numpy as np
import copy
import json
import os
import gym
import gym.wrappers
import gym.spaces
import gym.envs

from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def compute_si(values):
    """
    Computes the symmetric index for the input array

    Arguments:
        values: A Nx2xd array where d is num joints/objects and 2 is for left/right
    """
    l, r = np.array(values).transpose([1, 0, 2])
    xl = np.linalg.norm(l, axis=1).sum()
    xr = np.linalg.norm(r, axis=1).sum()
    return 200 * abs(xl - xr) / (xl + xr)


def calc_best_distance(l, r):
    """
    Calculates best distance with optimal shift (to get rid of phase difference)

    Arguments:
        l: A Nxd matrix
        r: A Nxd matrix
    """
    best_dist, best_shift = float("inf"), 0
    for shift in range(l.shape[0]):
        dist = np.linalg.norm(l - np.roll(r, shift), ord=1, axis=1).sum()
        if dist < best_dist:
            best_dist, best_shift = dist, shift

    return best_dist, best_shift


def compute_msi(values):
    """
    Computes the modified symmetric index for the input array

    Arguments:
        values: A Nx2xd array where d is num joints/objects and 2 is for left/right
    """
    l, r = np.array(values).transpose([1, 0, 2])

    w = 50 / (np.linalg.norm(l, ord=1, axis=1) + np.linalg.norm(r, ord=1, axis=1)).sum()

    distance, shift = calc_best_distance(l, r)

    return w * distance


class MetricsEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dt = getattr(self, "control_step", 1 / 60)
        # TODO assert PyBullet or similar

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        com_inds = [
            i
            for i, joint in enumerate(self.unwrapped.robot.ordered_joints)
            if "abdomen" in joint.joint_name
        ]
        self.left_joint_inds = [
            i
            for i, joint in enumerate(self.unwrapped.robot.ordered_joints)
            if "left" in joint.joint_name
        ]
        self.right_joint_inds = list(
            set(range(len(self.unwrapped.robot.ordered_joints)))
            - set(com_inds)
            - set(self.left_joint_inds)
        )
        self.prev_contact = True
        self.first_strike = True
        self.reset_metrics()
        return obs

    def reset_metrics(self):
        self.metrics = {"torque": [], "joint_angle": []}

    def integrate_metrics(self, info):
        side_contact = self.unwrapped.robot.feet_contact[0]
        strike = not self.prev_contact and side_contact
        self.prev_contact = side_contact

        if strike and len(self.metrics["torque"]) * self.dt > 0.5:
            if not self.first_strike:
                info["metrics"] = {
                    "torque": compute_si(self.metrics["torque"]),
                    "joint_angle": compute_msi(self.metrics["joint_angle"]),
                }
                # print(len(self.metrics["torque"]) * self.dt)
            self.first_strike = False
            self.reset_metrics()
        return info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.compute_side_metrics(action)
        info = self.integrate_metrics(info)
        return obs, rew, done, info

    def compute_side_metrics(self, action):
        action = np.array(action)
        self.metrics["torque"].append(
            [action[self.left_joint_inds], action[self.right_joint_inds]]
        )
        joint_angles = np.array(
            [j.get_position() for j in self.unwrapped.robot.ordered_joints]
        )
        self.metrics["joint_angle"].append(
            [joint_angles[self.left_joint_inds], joint_angles[self.right_joint_inds]]
        )


def average_values(array):
    max_len = max([len(stride) for stride in array])

    # fix the lengths (all will be stretched to `max_len`)
    array = [
        interp1d(range(len(arr)), arr)(np.linspace(0, len(arr) - 1, max_len))
        for arr in array
    ]

    # smooth out the signal and average over strides
    return (
        gaussian_filter1d(np.concatenate(array), sigma=5)
        .reshape((-1, max_len))
        .sum(axis=0)
    )


def phase_plot(ql, qdotl, qr, qdotr, skip_strides=0, render=True, save_path=None):
    ql = average_values(ql[skip_strides:])
    qdotl = average_values(qdotl[skip_strides:])
    qr = average_values(qr[skip_strides:])
    qdotr = average_values(qdotr[skip_strides:])

    distance = compute_msi(
        np.stack([np.stack([ql, qdotl]), np.stack([qr, qdotr])]).transpose((2, 0, 1))
    )

    if render or save_path:
        plt.ioff()
        plt.plot(ql, qdotl, "ro", markersize=1, label="left")
        plt.plot(qr, qdotr, "go", markersize=1, label="right")
        plt.xlabel("Hip Flexion Angle")
        plt.ylabel("Hip Flexion Velocity")

        if render:
            plt.show()
        if save_path:
            plt.savefig(save_path)

    return distance


class PhasePlotEnv(gym.Wrapper):
    def __init__(
        self, env, joint_name, save_path, strides=10, skip_strides=2, dt=1 / 480
    ):
        super().__init__(env)
        self.joint_name = joint_name
        self.save_path = save_path
        self.skip_strides = skip_strides
        self.strides = strides + skip_strides
        self.dt = dt

        while isinstance(env, gym.Wrapper):
            if isinstance(env, gym.wrappers.TimeLimit):
                env._max_episode_steps = float("inf")
            env = env.env

        # TODO assert PyBullet or similar

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self.unwrapped._p.setPhysicsEngineParameter(fixedTimeStep=self.dt)
        joints = [
            joint
            for joint in self.unwrapped.robot.ordered_joints
            if self.joint_name in joint.joint_name
        ]
        assert len(joints) == 2
        if "left" in joints[0].joint_name:
            self.left_joint, self.right_joint = joints
        else:
            self.right_joint, self.left_joint = joints
        self.prev_contact = True
        self.strike_num = 0
        self.steps_since = 0
        self.reset_metrics()
        return obs

    def reset_metrics(self):
        self.ql = [[] for _ in range(self.strides)]
        self.qdotl = [[] for _ in range(self.strides)]
        self.qr = [[] for _ in range(self.strides)]
        self.qdotr = [[] for _ in range(self.strides)]

    def integrate_metrics(self, info, done):
        side_contact = self.unwrapped.robot.feet_contact[0]
        strike = not self.prev_contact and side_contact
        self.prev_contact = side_contact
        self.steps_since += 1

        if strike and self.steps_since * self.dt > 0.5:
            self.steps_since = 0
            self.strike_num += 1
            if self.strike_num == self.strides:
                done = True

                distance = phase_plot(
                    self.ql,
                    self.qdotl,
                    self.qr,
                    self.qdotr,
                    skip_strides=self.skip_strides,
                    render=False,
                    save_path=os.path.join(self.save_path, "phase_plot.svg"),
                )
                info["metrics"] = {"phase_plot_index": distance}

        return info, done

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.compute_side_metrics(action)
        info, done = self.integrate_metrics(info, done)
        return obs, rew, done, info

    def compute_side_metrics(self, action):
        ql, qdotl = self.left_joint.get_state()
        qr, qdotr = self.right_joint.get_state()
        self.ql[self.strike_num].append(ql)
        self.qdotl[self.strike_num].append(qdotl)
        self.qr[self.strike_num].append(qr)
        self.qdotr[self.strike_num].append(qdotr)


if __name__ == "__main__":
    import pickle
    import sys

    if len(sys.argv) < 2:
        print("Please provide the states data path")
        sys.exit(1)

    with open(sys.argv[1], "rb") as fp:
        state = pickle.load(fp)

    clen = 1680
    steps = 20

    motor_pos = np.zeros((steps, clen, 10))
    motor_vel = np.zeros((steps, clen, 10))

    for i in range(steps):
        for j in range(clen):
            for k in range(10):
                motor_pos[i][j][k] = state[i * clen + j][0][k]
                motor_vel[i][j][k] = state[i * clen + j][1][k]

    save_dir = os.path.dirname(sys.argv[1])
    
    distance = phase_plot(
        motor_pos[:, :, 3],
        motor_vel[:, :, 3],
        motor_pos[:, :, 8],
        motor_vel[:, :, 8],
        render=False,
        save_path=os.path.join(save_dir, "phase_plot.svg"),
    )

    l = motor_pos[:, :, :5].reshape((-1, 5))
    r = motor_pos[:, :, 5:].reshape((-1, 5))
    metrics = {
        "torque": 0,
        "joint_angle": compute_si(np.stack([l, r]).transpose((1, 0, 2))),
        "phase_plot_index": distance,
    }

    print(metrics)
    with open(os.path.join(save_dir, "evaluate.json"), "w") as jfile:
        json.dump(metrics, jfile)



