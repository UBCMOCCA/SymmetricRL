from .env_utils import register, register_symmetric_envs


# Mirror_Walker2DBulletEnv-v0
# Symmetric_Walker2DBulletEnv-v0 (only used for net)
# SymmetricV2_Walker2DBulletEnv-v0 (only used for net2)
# Phase_Walker2DBulletEnv-v0 (only used for phase-based)
register_symmetric_envs(
    "pybullet_envs:Walker2DBulletEnv-v0",
    gait_cycle_length=0.8,
    dt=1 / 60,
    mirror_inds={
        #### observation:
        "com_obs_inds": [0, 2, 3, 5, 7],
        "neg_obs_inds": [1, 4, 6],
        "left_obs_inds": list(range(8, 14)) + [20],
        "right_obs_inds": list(range(14, 20)) + [21],
        "sideneg_obs_inds": [],
        #### action:
        "com_act_inds": [],
        "neg_act_inds": [],
        "sideneg_act_inds": [],
        "left_act_inds": list(range(0, 3)),
        "right_act_inds": list(range(3, 6)),
    },
)

# Mirror_HumanoidBulletEnv-v0
# Symmetric_HumanoidBulletEnv-v0 (only used for net)
# SymmetricV2_HumanoidBulletEnv-v0 (only used for net2)
# Phase_HumanoidBulletEnv-v0 (only used for phase-based)
register_symmetric_envs(
    "pybullet_envs:HumanoidBulletEnv-v0",
    gait_cycle_length=1,
    dt=1 / 60,
    mirror_inds={
        #### observation:
        "com_obs_inds": [
            0,  # z
            2,  # cos(yaw)
            3,  # vx
            5,  # vz
            7,  # pitch
            # common joints
            10,
            11,
        ],
        "neg_obs_inds": [
            1,  # sin(yaw)
            4,  # vy
            6,  # roll
            # neg joints
            8,
            9,
            12,
            13,
        ],
        "left_obs_inds": list(range(22, 30)) + list(range(36, 42)) + [43],
        "right_obs_inds": list(range(14, 22)) + list(range(30, 36)) + [42],
        "sideneg_obs_inds": list(range(30, 34)),
        #### action:
        "com_act_inds": [1],
        "neg_act_inds": [0, 2],
        "left_act_inds": [7, 8, 9, 10, 14, 15, 16],
        "right_act_inds": [3, 4, 5, 6, 11, 12, 13],
        "sideneg_act_inds": [11, 12],
    },
)

# Mirror_Walker3DCustomEnv-v0
# Symmetric_Walker3DCustomEnv-v0 (only used for net)
# SymmetricV2_Walker3DCustomEnv-v0 (only used for net2)
# Phase_Walker3DCustomEnv-v0 (only used for phase-based)
register_symmetric_envs(
    "mocca_envs:Walker3DCustomEnv-v0",
    gait_cycle_length=1,
    dt=1 / 60,
    mirror_inds={
        #### observation:
        "com_obs_inds": [0, 1, 3, 5, 7, 51, 28],
        "left_obs_inds": [
            14,
            15,
            16,
            17,
            18,
            23,
            24,
            25,
            26,
            35,
            36,
            37,
            38,
            39,
            44,
            45,
            46,
            47,
            49,
        ],
        "right_obs_inds": [
            9,
            10,
            11,
            12,
            13,
            19,
            20,
            21,
            22,
            30,
            31,
            32,
            33,
            34,
            40,
            41,
            42,
            43,
            48,
        ],
        "neg_obs_inds": [2, 4, 6, 8, 27, 29, 50],
        "sideneg_obs_inds": [],
        #### action:
        "com_act_inds": [1],
        "neg_act_inds": [0, 2],
        "left_act_inds": [8, 9, 10, 11, 12, 17, 18, 19, 20],
        "right_act_inds": [3, 4, 5, 6, 7, 13, 14, 15, 16],
        "sideneg_act_inds": [],
    },
)


# Mirror_Walker3DStepperEnv-v0
# Symmetric_Walker3DStepperEnv-v0 (only used for net)
# SymmetricV2_Walker3DStepperEnv-v0 (only used for net2)
# Phase_Walker3DStepperEnv-v0 (only used for phase-based)
register_symmetric_envs(
    "mocca_envs:Walker3DStepperEnv-v0",
    gait_cycle_length=1,
    dt=1 / 60,
    mirror_inds={
        #### observation:
        "com_obs_inds": [0, 1, 3, 5, 7, 28, 51, 52, 54, 56, 57, 59],
        "left_obs_inds": [
            14,
            15,
            16,
            17,
            18,
            23,
            24,
            25,
            26,
            35,
            36,
            37,
            38,
            39,
            44,
            45,
            46,
            47,
            49,
        ],
        "right_obs_inds": [
            9,
            10,
            11,
            12,
            13,
            19,
            20,
            21,
            22,
            30,
            31,
            32,
            33,
            34,
            40,
            41,
            42,
            43,
            48,
        ],
        "neg_obs_inds": [2, 4, 6, 8, 27, 29, 50, 53, 55, 58],
        "sideneg_obs_inds": [],
        #### action:
        "com_act_inds": [1],
        "neg_act_inds": [0, 2],
        "left_act_inds": [8, 9, 10, 11, 12, 17, 18, 19, 20],
        "right_act_inds": [3, 4, 5, 6, 7, 13, 14, 15, 16],
        "sideneg_act_inds": [],
    },
)

# Mirror_CassiePhaseMocca2DEnv-v0
# Symmetric_CassiePhaseMocca2DEnv-v0 (only used for net)
# SymmetricV2_CassiePhaseMocca2DEnv-v0 (only used for net2)
register_symmetric_envs(
    "mocca_envs:CassiePhaseMocca2DEnv-v0",
    mirror_inds={
        "neg_obs_inds": [
            # y
            0,
            # quat x
            3,
            # quat z
            5,
            # y velocity
            21,
            # x angular speed
            23,
            # z angular speed
            25,
        ],
        "sideneg_obs_inds": [
            # left abduction
            6,
            # left yaw
            7,
            # left abduction speed
            26,
            # left yaw speed
            27,
        ],
        "com_obs_inds": [1, 2, 4, 20, 22, 24],
        "left_obs_inds": list(range(6, 13)) + list(range(26, 33)) + [40],
        "right_obs_inds": list(range(13, 20)) + list(range(33, 40)) + [41],
        # action:
        "com_act_inds": [],
        "left_act_inds": list(range(0, 5)),
        "right_act_inds": list(range(5, 10)),
        "neg_act_inds": [],
        "sideneg_act_inds": [0, 1],
    },
)

# Phase_CassiePhaseMocca2DEnv-v0 (only used for phase-based)
register(
    id="Phase_CassiePhaseMocca2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassiePhaseMirrorEnv",
    max_episode_steps=1000,
    kwargs={"planar": True},
)
