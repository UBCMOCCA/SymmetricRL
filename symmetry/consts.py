from common.misc_utils import (
    linear_decay,
    exponential_decay,
    set_optimizer_lr,
    StringEnum,
)

MirrorMethods = StringEnum(["none", "net", "traj", "loss", "phase", "net2"])
