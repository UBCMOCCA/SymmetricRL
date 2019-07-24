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

MIN_GAIT_LEN = 0.55


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
    Calculates best L-1 distance between two matrices with the optimal shift (to get rid of the phase difference)

    Arguments:
        l: A Nxd matrix
        r: A Nxd matrix
    """
    best_dist, best_shift = float("inf"), 0
    for shift in range(l.shape[0]):
        dist = np.linalg.norm(l - np.roll(r, shift), ord=1, axis=1).mean()
        if dist < best_dist:
            best_dist, best_shift = dist, shift

    return best_dist, best_shift


def compute_msi(values):
    """
    Computes the modified symmetric index for the input array.
    The array is scaled by the maximum absolute value to make the reuslts scale-invariant.

    Arguments:
        values: A Nx2xd array where d is num joints/objects and 2 is for left/right
    """
    # scale by max |v| in each dimension to make the results scale-invariant
    scale = np.abs(values).max(axis=0).max(axis=0)
    l, r = np.array(values).transpose([1, 0, 2]) / scale

    distance, shift = calc_best_distance(l, r)

    return 2 * distance


def average_values(arrays):
    """
    Takes in a list of lists. It does two things:
      1- makes all the inner lists have the same sizes (using interpolation)
      2- averages the values
    
    The output has the same length as the longest array in the input list

    Arguments:
        arrays: list of list of number
    """
    max_len = max([len(stride) for stride in arrays])

    # fix the lengths (all will be stretched to `max_len`)
    arrays = [
        interp1d(range(len(arr)), arr)(np.linspace(0, len(arr) - 1, max_len))
        for arr in arrays
    ]

    # smooth out the signal and average over strides
    return (
        gaussian_filter1d(np.concatenate(arrays), sigma=5)
        .reshape((-1, max_len))
        .sum(axis=0)
    )


def phase_plot(ql, qdotl, qr, qdotr, skip_strides=0, render=True, save_path=None):
    """
    Computes the phase-plot index (PPI) and draws the phase-plot

    Arguments:
        ql, qdotl: position and velocities for the left joint
        qr, qdotr: position and velocities for the right joint

        render: If True will render the plot to the screen (waits for the plot to be closed)
        save_path: If provided the plot is saved to the specified location
        
        skip_strides: If provided ignores the first few strides and computes everything using the rest of the data
    """
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


class MetricsEnv(gym.Wrapper):
    """
    A wrapper that computes the following metrics:
      - Actuation Symmetric Index (ASI)
      - Phase-Plot Index (PPI)
    
    It also changes the control time-step of the said environment to get a higher resolution phase-plot.
    
    The results are returned in the extras part of the `step` function under the name "metrics":
    ```
        obs, rew, done, extras = step(action)
        if "metrics" in extras:
            print("Metrics:", extras["metrics"])
    ```
    
    This wrapper only works on PyBullet/mocca_envs environments. Tested on:
      - Walker2D
      - Walker3DCustomEnv
      - Walker3DStepperEnv
    """

    def __init__(
        self, env, pp_joint_name, save_path, strides=10, skip_strides=2, dt=1 / 480
    ):
        super().__init__(env)
        assert hasattr(self.unwrapped, "_p")

        self.pp_joint_name = pp_joint_name
        self.save_path = save_path
        self.skip_strides = skip_strides
        self.strides = strides + skip_strides
        self.dt = dt

        if hasattr(self.unwrapped, "evaluation_mode"):
            self.unwrapped.evaluation_mode()

        while isinstance(env, gym.Wrapper):
            if isinstance(env, gym.wrappers.TimeLimit):
                env._max_episode_steps = float("inf")
            env = env.env

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self.unwrapped._p.setPhysicsEngineParameter(fixedTimeStep=self.dt)

        joints = np.array(self.unwrapped.robot.ordered_joints)

        self.left_joint_inds = [
            i for i, j in enumerate(joints) if "left" in j.joint_name
        ]
        self.right_joint_inds = [
            i
            for i, j in enumerate(joints)
            if "left" not in j.joint_name and "abdomen" not in j.joint_name
        ]

        self.torque_limits = np.array(
            [
                self.unwrapped.robot.power * j.power_coef
                if hasattr(j, "power_coef")
                else j.torque_limit
                for j in joints[self.left_joint_inds]
            ]
        )

        self.pp_left_joint = [
            j
            for j in joints[self.left_joint_inds]
            if self.pp_joint_name in j.joint_name
        ][0]
        self.pp_right_joint = [
            j
            for j in joints[self.right_joint_inds]
            if self.pp_joint_name in j.joint_name
        ][0]

        self.prev_contact = True
        self.strike_num = 0
        self.steps_since = 0

        self.torques = []
        self.ql = [[] for _ in range(self.strides)]
        self.qdotl = [[] for _ in range(self.strides)]
        self.qr = [[] for _ in range(self.strides)]
        self.qdotr = [[] for _ in range(self.strides)]

        return obs

    def integrate_readings(self, info, done):

        side_contact = self.unwrapped.robot.feet_contact[0]
        strike = not self.prev_contact and side_contact
        self.prev_contact = side_contact
        self.steps_since += 1

        if strike and self.steps_since * self.dt > MIN_GAIT_LEN:
            self.steps_since = 0
            self.strike_num += 1

            if self.strike_num > self.skip_strides:
                info["metrics"] = {"torque": compute_si(self.torques)}

            self.torques = []

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
        self.compute_side_readings(action)
        info, done = self.integrate_readings(info, done)
        return obs, rew, done, info

    def compute_side_readings(self, action):
        action = np.array(action)
        self.torques.append(
            [
                action[self.left_joint_inds] * self.torque_limits,
                action[self.right_joint_inds] * self.torque_limits,
            ]
        )

        ql, qdotl = self.pp_left_joint.get_state()
        qr, qdotr = self.pp_right_joint.get_state()
        self.ql[self.strike_num].append(ql)
        self.qdotl[self.strike_num].append(qdotl)
        self.qr[self.strike_num].append(qr)
        self.qdotr[self.strike_num].append(qdotr)


def compute_metrics_for_cassie():
    import pickle
    import sys

    clen = 1680
    steps = 20

    if len(sys.argv) < 2:
        print("Please provide the states data path")
        sys.exit(1)

    with open(sys.argv[1][: -len("_torque")], "rb") as fp:
        state = pickle.load(fp)

    with open(sys.argv[1], "rb") as fp:
        torques = np.array(pickle.load(fp)).reshape((steps, clen, 2, 5))

    torque_si = np.mean([compute_si(step_toques) for step_toques in torques])

    motor_pos = np.zeros((steps, clen, 10))
    motor_vel = np.zeros((steps, clen, 10))

    for i in range(steps):
        for j in range(clen):
            for k in range(10):
                motor_pos[i][j][k] = state[i * clen + j][0][k]
                motor_vel[i][j][k] = state[i * clen + j][1][k]

    save_dir = os.path.dirname(sys.argv[1])

    distance = phase_plot(
        motor_pos[:, :, 2],
        motor_vel[:, :, 2],
        motor_pos[:, :, 7],
        motor_vel[:, :, 7],
        render=False,
        save_path=os.path.join(save_dir, "phase_plot.svg"),
    )

    l = motor_pos[:, :, :5].reshape((-1, 5))
    r = motor_pos[:, :, 5:].reshape((-1, 5))
    metrics = {"torque": torque_si, "phase_plot_index": distance}

    print(metrics)
    with open(os.path.join(save_dir, "evaluate.json"), "w") as jfile:
        json.dump(metrics, jfile)


if __name__ == "__main__":
    compute_metrics_for_cassie()
