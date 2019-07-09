import torch as th
import numpy as np
import copy
import os
import gym
import gym.spaces
import gym.envs


class SymmetricEnv(gym.Wrapper):
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

    def __init__(self, env, minds=None):
        super().__init__(env)

        if isinstance(env, gym.Wrapper):
            env = env.unwrapped

        if minds is None:
            minds = env.mirror_indices

        assert len(minds["left_obs_inds"]) == len(minds["right_obs_inds"])
        assert len(minds["left_act_inds"]) == len(minds["right_act_inds"])
        # *_in
        ci = len(minds.get("com_obs_inds", []))
        ni = len(minds.get("neg_obs_inds", []))
        si = len(minds.get("left_obs_inds", []))
        # *_out
        co = len(minds.get("com_act_inds", []))
        no = len(minds.get("neg_act_inds", []))
        so = len(minds.get("left_act_inds", []))

        # print(ci, ni, si, co, no, so)
        # make sure the sizes match the observation space
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert (ci + ni + 2 * si) == env.observation_space.shape[0]
        assert (co + no + 2 * so) == env.action_space.shape[0]

        env.sym_act_inds = [co, no, so]

        high = np.ones(2 * env.observation_space.shape[0]) * np.inf
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        # high = np.ones(co + no + so) * np.inf
        # self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # observation indices
        self.neg_obs_inds = minds.get("sideneg_obs_inds", []) + minds.get(
            "neg_obs_inds", []
        )
        self.lr_obs_inds = minds["left_obs_inds"] + minds["right_obs_inds"]
        self.rl_obs_inds = minds["right_obs_inds"] + minds["left_obs_inds"]
        # action indices
        self.sideneg_act_inds = minds.get("sideneg_act_inds", [])
        self.reverse_act_inds = (
            minds.get("com_act_inds", [])
            + minds.get("neg_act_inds", [])
            + minds.get("left_act_inds", [])
            + minds.get("right_act_inds", [])
        )
        # for `common.envs_utils.get_mirror_function`
        env.get_mirror_indices = lambda: [
            self.neg_obs_inds,
            minds["right_obs_inds"],
            minds["left_obs_inds"],
            minds.get("neg_act_inds", []) + minds.get("sideneg_act_inds", []),
            minds.get("right_act_inds", []),
            minds.get("left_act_inds", []),
        ]

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

    def render(self, *args, **kwargs):
        # print(args, kwargs)
        self.env.render(*args, **kwargs)

    def replace_wrapped_env(self, env):
        # TODO: find a fix not to do this ...
        return self.env.replace_wrapped_env(env)


def register(id, **kvargs):
    if id in gym.envs.registration.registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, **kvargs)


def register_symmetric_env(env_id, mirror_inds):
    def make_sym_env(*args, **kwargs):
        return SymmetricEnv(env=gym.make(env_id, *args, **kwargs), minds=mirror_inds)

    new_id = "Symmetric_%s" % env_id.split(":")[-1]
    register(id=new_id, entry_point=make_sym_env)
    return new_id
