from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter
seg = "lib.train.dataset.%s"

dataset_dict = dict(
    hsitrack=DatasetInfo(module=pt % "hsitrack", class_name="HSITrackDataset", kwargs=dict(split='test')),
    musthsi=DatasetInfo(module=pt % "musthsi", class_name="MUSTHSIDataset", kwargs=dict(split='test')),
    # 与 musthsi 相同类，但读 MUSTHSI 根目录下的 train/（list.txt + 各序列），用于看训练集序列上的跟踪表现
    musthsi_train=DatasetInfo(module=pt % "musthsi", class_name="MUSTHSIDataset", kwargs=dict(split='train')),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset