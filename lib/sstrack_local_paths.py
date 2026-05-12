"""
集中解析 SSTrack 在本机上的路径，换服务器时优先用环境变量覆盖，否则落在仓库下的 outputs/。

可选环境变量（未设置则用括号内默认值，均支持 ~ 展开）：
  SSTRACK_WORKSPACE          训练 checkpoint / 日志等工作目录（默认 <repo>/outputs）
  SSTRACK_TENSORBOARD_DIR    TensorBoard（默认 <workspace>/tensorboard）
  SSTRACK_PRETRAINED_NETWORKS  预训练权重目录（默认 <workspace>/pretrained_networks）
  SSTRACK_MUSTHSI_DIR        MUST-HSI 数据集根目录
  SSTRACK_HSITRACK_DIR       HSITrack 数据集根目录
  SSTRACK_PRJ_DIR            含 experiments/ 的代码根（默认 <repo>）
  SSTRACK_SAVE_DIR           测试时默认 checkpoint 根（默认 <workspace>/save）
  SSTRACK_TRACKING_RESULTS   测试结果目录（默认 <workspace>/test/tracking_results）
  SSTRACK_TEST_NETWORKS      测试用网络缓存（默认 <workspace>/test/networks）
  SSTRACK_RESULT_PLOTS       结果图（默认 <workspace>/test/result_plots）
  SSTRACK_SEGMENTATION_RESULTS 分割结果（默认 <workspace>/test/segmentation_results）
"""
import os


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _env_path(name, default):
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    return os.path.abspath(os.path.expanduser(str(v).rstrip('/')))


def workspace_dir():
    return _env_path('SSTRACK_WORKSPACE', os.path.join(repo_root(), 'outputs'))


def tensorboard_dir():
    w = workspace_dir()
    return _env_path('SSTRACK_TENSORBOARD_DIR', os.path.join(w, 'tensorboard'))


def pretrained_networks_dir():
    w = workspace_dir()
    return _env_path('SSTRACK_PRETRAINED_NETWORKS', os.path.join(w, 'pretrained_networks'))


def musthsi_dir():
    return _env_path('SSTRACK_MUSTHSI_DIR', '/data3/PublicDataset/Public/MUST-BIT/HSICV/')


def hsitrack_dir():
    return _env_path('SSTRACK_HSITRACK_DIR', '/data/users/fengtao/SSTrack/data/HSICV/')


def prj_dir():
    return _env_path('SSTRACK_PRJ_DIR', repo_root())


def save_dir():
    w = workspace_dir()
    return _env_path('SSTRACK_SAVE_DIR', os.path.join(w, 'save'))


def test_tracking_results():
    w = workspace_dir()
    return _env_path('SSTRACK_TRACKING_RESULTS', os.path.join(w, 'test', 'tracking_results'))


def test_networks():
    w = workspace_dir()
    return _env_path('SSTRACK_TEST_NETWORKS', os.path.join(w, 'test', 'networks'))


def test_result_plots():
    w = workspace_dir()
    return _env_path('SSTRACK_RESULT_PLOTS', os.path.join(w, 'test', 'result_plots'))


def test_segmentation_results():
    w = workspace_dir()
    return _env_path('SSTRACK_SEGMENTATION_RESULTS', os.path.join(w, 'test', 'segmentation_results'))
