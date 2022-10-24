import numpy as np
import json
import os
import h5py
import collections

from .data import Split


def get_range(imgs):
    r = np.any(imgs.reshape(imgs.shape[0], -1), axis=-1).nonzero()
    return np.min(r), np.max(r)


class Hdf5Reader:
    def __init__(self, path):
        try:
            self.file = h5py.File(path, "r")
        except Exception:
            print('{} not found!'.format(path))
            self.file = None

    def __getitem__(self, key):
        data = {'id': key}
        if self.file is None:
            return data
        group = self.file[key]
        for k in group:
            data[k] = group[k]
        return data


class FileManager:
    def __init__(self, files):
        self.files = {}
        for k, v in files.items():
            self.files[k] = Hdf5Reader(v["path"])

    def __getitem__(self, key):
        p = key.find('/')
        if key[:p] in self.files:
            ret = self.files[key[:p]][key[p+1:]]
            ret['id'] = key.replace('/', '-')
            return ret
        elif '/' in self.files:
            ret = self.files['/'][key]
            ret['id'] = key.replace('/', '-')
            return ret
        else:
            raise KeyError('{} not found'.format(key))



class Dataset:
    def __init__(self, split_path, affine=False, mask=False, paired=False, task=None, batch_size=None):
        with open(split_path, 'r') as f:
            config = json.load(f)
        self.files = FileManager(config['files'])
        self.subset = {}

        for k, v in config['subsets'].items():
            self.subset[k] = {}
            for entry in v:
                self.subset[k][entry] = self.files[k][entry]

        self.paired = paired

        def convert_int(key):
            try:
                return int(key)
            except ValueError as e:
                return key
        self.schemes = dict([(convert_int(k), v)
                             for k, v in config['schemes'].items()])
        self.image_size = config.get("image_size", [160, 160, 160])
        self.segmentation_class_value = config.get(
            'segmentation_class_value', None)

        
        for k, v in self.subset.items():
            print('Number of data in {} is {}'.format(k, len(v)))

        self.task = task
        if self.task is None:
            self.task = config.get("task", "registration")
        if not isinstance(self.task, list):
            self.task = [self.task]

        self.batch_size = batch_size

    def get_pairs_adj(self, data):
        pairs = []
        d1 = None
        for d2 in data:
            if d1 is None:
                d1 = d2
            else:
                pairs.append((d1, d2))
                pairs.append((d2, d1))
                d1 = None
                
        return pairs

    def get_pairs(self, data, ordered=True):
        pairs = []
        for i, d1 in enumerate(data):
            for j, d2 in enumerate(data):
                if i != j:
                    if ordered or i < j:
                        pairs.append((d1, d2))
        return pairs

    def generate_pairs(self, arr, loop=False, adj = False):
        if self.paired or adj:
            sets = self.get_pairs_adj(arr)
        else:
            sets = self.get_pairs(arr, ordered=True)

        while True:
            if loop:
                np.random.shuffle(sets)
            for d1, d2 in sets:
                yield (d1, d2)
            if not loop:
                break

    def generator(self, subset, batch_size=None, loop=False, adj = False):
        if batch_size is None:
            batch_size = self.batch_size
        valid_mask = np.ones([2], dtype=np.bool)
        scheme = self.schemes[subset]
        if 'registration' in self.task:
            generators = [(self.generate_pairs(list(self.subset[k].values()), loop, adj))
                          for k, fraction in scheme.items()]
            fractions = [int(np.round(fraction * batch_size))
                         for k, fraction in scheme.items()]

            while True:
                imgs = [batch_size] + self.image_size + [1]
                ret = dict()
                ret['voxelT1'] = np.zeros(imgs, dtype=np.float32)
                ret['voxelT2'] = np.zeros(imgs, dtype=np.float32)
                ret['atlasT1'] = np.zeros(imgs, dtype=np.float32)
                ret['atlasT2'] = np.zeros(imgs, dtype=np.float32)

                ret['seg1'] = np.zeros(imgs, dtype=np.float32)
                ret['seg2'] = np.zeros(imgs, dtype=np.float32)
                ret['point1'] = np.ones(
                    (batch_size, np.sum(valid_mask), 3), dtype=np.float32) * (-1)
                ret['point2'] = np.ones(
                    (batch_size, np.sum(valid_mask), 3), dtype=np.float32) * (-1)
                ret['id1'] = np.empty((batch_size), dtype='<U30')
                ret['id2'] = np.empty((batch_size), dtype='<U30')

                i = 0
                flag = True
                cc = collections.Counter(np.random.choice(range(len(fractions)), size=[
                                            batch_size, ], replace=True, p=fractions))
                nums = [cc[i] for i in range(len(fractions))]
                for gen, num in zip(generators, nums):
                    assert not self.paired or num % 2 == 0
                    for t in range(num):
                        try:
                            while True:
                                d1, d2 = next(gen)
                                break
                        except StopIteration:
                            flag = False
                            break

                        if 'segmentation' in d1:
                            ret['seg1'][i, ..., 0] = d1['segmentation']
                        if 'segmentation' in d2:
                            ret['seg2'][i, ..., 0] = d2['segmentation']
                        if 'point' in d1:
                            ret['point1'][i] = d1['point'][...][valid_mask]
                        if 'point' in d2:
                            ret['point2'][i] = d2['point'][...][valid_mask]

                        ret['voxelT1'][i, ..., 0], ret['atlasT1'][i, ...,0] = d1['volumeT1'], d2['volumeT1']
                        ret['voxelT2'][i, ..., 0], ret['atlasT2'][i, ...,0] = d1['volumeT2'], d2['volumeT2']

                        ret['id1'][i] = d1['id']
                        ret['id2'][i] = d2['id']
                        i += 1

                if flag:
                    assert i == batch_size
                    yield ret
                else:
                    yield ret
                    break