import torch as th
import numpy as np
import copy
import os
import gym
import gym.spaces
import gym.envs

from .consts import MirrorMethods


class MirrorIndicesEnv(gym.Wrapper):
    """
        :params mirror_indices: indices used for mirroring the environment
            example:
            mirror_indices = {
                #### observation:
                "com_obs_inds": [],          # common indices (in the observation)
                "left_obs_inds": [],         # indices of the right side (in the observation)
                "right_obs_inds": [],        # indices of the left side (in the observation)
                "neg_obs_inds": [],          # common indices that should be negated (in the observation)
                "sideneg_obs_inds": [],      # side indices that should be negated (in the observation)

                #### action:
                "com_act_inds": [],          # common indices of the action
                "neg_act_inds": [],          # common indices of the action that should be negated when mirrored
                "left_act_inds": [],         # indices of the left side in the action
                "right_act_inds": [],        # indices of the right side in the action
                "sideneg_act_inds": [],      # indices of the side that should be negated
            }


        ** Please do not use numpy arrays instead of list for the indices

        ** Point of difference with MirrorEnv:
          - both neg obs indices are treated the same here
    """

    def __init__(self, env, minds):
        super().__init__(env)
        self.minds = minds

        env = self.unwrapped

        assert len(minds["left_obs_inds"]) == len(minds["right_obs_inds"])
        assert len(minds["left_act_inds"]) == len(minds["right_act_inds"])
        # *_in
        ci = len(minds["com_obs_inds"])
        ni = len(minds["neg_obs_inds"])
        si = len(minds["left_obs_inds"])
        # *_out
        co = len(minds["com_act_inds"])
        no = len(minds["neg_act_inds"])
        so = len(minds["left_act_inds"])

        self.in_sizes = ci, ni, si
        self.out_sizes = co, no, so

        # print(ci, ni, si, co, no, so)
        # make sure the sizes match the observation space
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert (ci + ni + 2 * si) == env.observation_space.shape[0]
        assert (co + no + 2 * so) == env.action_space.shape[0]

        # for `common.envs_utils.get_mirror_function`
        env.get_mirror_indices = lambda: [
            minds["sideneg_obs_inds"] + minds["neg_obs_inds"],
            minds["right_obs_inds"],
            minds["left_obs_inds"],
            minds["neg_act_inds"] + minds["sideneg_act_inds"],
            minds["right_act_inds"],
            minds["left_act_inds"],
        ]

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)


class SymmetricEnv(MirrorIndicesEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        env = self.unwrapped
        minds = self.minds

        env.sym_act_inds = self.out_sizes  # co, no, so

        high = np.ones(2 * env.observation_space.shape[0]) * np.inf
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # observation indices
        self.neg_obs_inds = minds["sideneg_obs_inds"] + minds["neg_obs_inds"]
        self.lr_obs_inds = minds["left_obs_inds"] + minds["right_obs_inds"]
        self.rl_obs_inds = minds["right_obs_inds"] + minds["left_obs_inds"]
        # action indices
        self.sideneg_act_inds = minds["sideneg_act_inds"]
        self.reverse_act_inds = (
            minds["com_act_inds"]
            + minds["neg_act_inds"]
            + minds["left_act_inds"]
            + minds["right_act_inds"]
        )
        # should not use `common.envs_utils.get_mirror_function` on this environment
        # TODO: we can allow this later
        env.get_mirror_indices = None

    def reset(self, **kwargs):
        return self.fix_obs(self.env.reset(**kwargs))

    def step(self, act_):
        action = 0 * act_
        action[self.reverse_act_inds] = act_
        action[self.sideneg_act_inds] *= -1
        obs, reward, done, info = self.env.step(action)
        return self.fix_obs(obs), reward, done, info

    def fix_obs(self, obs):
        obs_m = np.array(obs)
        obs_m[self.neg_obs_inds] *= -1
        obs_m[self.lr_obs_inds] = obs_m[self.rl_obs_inds]
        return np.concatenate([obs, obs_m])


class PhaseSymmetryEnv(MirrorIndicesEnv):
    def __init__(self, gait_cycle_length=1.5, dt=1 / 60, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gait_cycle_length = gait_cycle_length
        self.ts_phase_increment = dt / gait_cycle_length

        env = self.unwrapped
        minds = self.minds

        high = np.ones(1 + env.observation_space.shape[0]) * np.inf
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # observation indices
        self.neg_obs_inds = minds["sideneg_obs_inds"] + minds["neg_obs_inds"]
        self.lr_obs_inds = minds["left_obs_inds"] + minds["right_obs_inds"]
        self.rl_obs_inds = minds["right_obs_inds"] + minds["left_obs_inds"]
        # action indices
        self.neg_act_inds = minds["sideneg_act_inds"] + minds["neg_act_inds"]
        self.lr_act_inds = minds["left_act_inds"] + minds["right_act_inds"]
        self.rl_act_inds = minds["right_act_inds"] + minds["left_act_inds"]

        # should not use `common.envs_utils.get_mirror_function` on this environment
        # TODO: we can allow this later
        env.get_mirror_indices = None

    def reset(self, **kwargs):
        self.phase = 0 if self.np_random.uniform() < 0.5 else 0.5
        return self.fix_obs(self.env.reset(**kwargs))

    def step(self, action):
        self.phase = (self.phase + self.ts_phase_increment) % self.gait_cycle_length
        if self.phase >= 0.5:
            action[self.neg_act_inds] *= -1
            action[self.lr_act_inds] = action[self.rl_act_inds]
        obs, reward, done, info = self.env.step(action)
        return self.fix_obs(obs), reward, done, info

    def fix_obs(self, obs):
        if self.phase >= 0.5:
            obs[self.neg_obs_inds] *= -1
            obs[self.lr_obs_inds] = obs[self.rl_obs_inds]
        return np.concatenate([[self.phase], obs])


def register(id, **kvargs):
    if id in gym.envs.registration.registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, **kvargs)


def register_symmetric_envs(env_id, mirror_inds, gait_cycle_length=None, dt=None):
    env_name = env_id.split(":")[-1]

    def make_mirror_env(*args, **kwargs):
        return MirrorIndicesEnv(
            env=gym.make(env_id, *args, **kwargs), minds=mirror_inds
        )

    def make_sym_env(*args, **kwargs):
        return SymmetricEnv(env=gym.make(env_id, *args, **kwargs), minds=mirror_inds)

    def make_phase_env(*args, **kwargs):
        return PhaseSymmetryEnv(
            env=gym.make(env_id, *args, **kwargs),
            minds=mirror_inds,
            gait_cycle_length=gait_cycle_length,
            dt=dt,
        )

    register(id="Mirror_%s" % env_name, entry_point=make_mirror_env)
    register(id="Symmetric_%s" % env_name, entry_point=make_sym_env)
    if gait_cycle_length is not None and dt is not None:
        register(id="Phase_%s" % env_name, entry_point=make_phase_env)


def get_env_name_for_method(env_name, mirror_method):
    if mirror_method == MirrorMethods.net:
        env_name = "Symmetric_" + env_name.split(":")[-1]
    elif mirror_method == MirrorMethods.loss or mirror_method == MirrorMethods.traj:
        env_name = "Mirror_" + env_name.split(":")[-1]
    elif mirror_method == MirrorMethods.phase:
        env_name = "Phase_" + env_name.split(":")[-1]
    return env_name
