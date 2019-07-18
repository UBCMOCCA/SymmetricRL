import torch as th
import numpy as np
import copy
import os
import gym
import gym.spaces
import gym.envs


def compute_si(values):
    """
    Computes the symmetric index for the input array

    Arguments:
        values: A Nx2xd array where the middle index is for left/right and the last index is for different joints/objects
    """
    l, r = np.array(values).transpose([1, 0, 2])
    l = np.sort(l, axis=0)
    r = np.sort(r, axis=0)
    return 200 * np.abs(np.subtract(l, r)).mean() / (np.abs(l) + np.abs(r)).mean()


class MetricsEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dt = getattr(self, "control_step", 1 / 60)
        # TODO assert PyBullet or similar

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        com_inds = [
            i
            for i, joint in enumerate(self.env.unwrapped.robot.ordered_joints)
            if "abdomen" in joint.joint_name
        ]
        self.left_joint_inds = [
            i
            for i, joint in enumerate(self.env.unwrapped.robot.ordered_joints)
            if "left" in joint.joint_name
        ]
        self.right_joint_inds = list(
            set(range(len(self.env.unwrapped.robot.ordered_joints)))
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
        side_contact = self.env.unwrapped.robot.feet_contact[0]
        strike = not self.prev_contact and side_contact
        self.prev_contact = side_contact

        if strike and len(self.metrics["torque"]) * self.dt > 0.5:
            if not self.first_strike:
                info["metrics"] = {k: compute_si(v) for k, v in self.metrics.items()}
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
            [j.get_position() for j in self.env.unwrapped.robot.ordered_joints]
        )
        self.metrics["joint_angle"].append(
            [joint_angles[self.left_joint_inds], joint_angles[self.right_joint_inds]]
        )
