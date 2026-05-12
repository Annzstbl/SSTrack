import os
import sys

_prj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _prj not in sys.path:
    sys.path.insert(0, _prj)

from lib.test.evaluation.environment import EnvSettings
from lib.sstrack_local_paths import (
    prj_dir,
    save_dir,
    musthsi_dir,
    hsitrack_dir,
    test_tracking_results,
    test_networks,
    test_result_plots,
    test_segmentation_results,
)


def local_env_settings():
    settings = EnvSettings()

    settings.musthsi_path = musthsi_dir()
    settings.hsitrack_path = hsitrack_dir()
    settings.prj_dir = prj_dir()
    settings.save_dir = save_dir()
    settings.network_path = test_networks()
    settings.result_plot_path = test_result_plots()
    settings.results_path = test_tracking_results()
    settings.segmentation_path = test_segmentation_results()

    return settings
