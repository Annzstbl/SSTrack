import os
import sys

_prj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _prj not in sys.path:
    sys.path.insert(0, _prj)

from lib.sstrack_local_paths import (
    workspace_dir,
    tensorboard_dir,
    pretrained_networks_dir,
    hsitrack_dir,
    musthsi_dir,
)


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = workspace_dir()
        self.tensorboard_dir = tensorboard_dir()
        self.pretrained_networks = pretrained_networks_dir()
        self.hsitrack_dir = hsitrack_dir()
        self.musthsi_dir = musthsi_dir()
