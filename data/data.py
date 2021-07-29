""" CNN/DM dataset"""
import json
import re
import os
from os.path import join

from torch.utils.data import Dataset


class CnnDmDataset(Dataset):
    def __init__(self, split: str, path: str, mono_abs: bool) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self._n_data, self._l_data = _count_data(self._data_path)
        self._mono_abs = mono_abs

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        #with open(join(self._data_path, '{}.json'.format(i))) as f:
        with open(join(self._data_path, self._l_data[i])) as f:
            js = json.loads(f.read())
        return js


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    l_data = list(filter(match, names))
    n_data = len(l_data)
    return n_data, l_data
