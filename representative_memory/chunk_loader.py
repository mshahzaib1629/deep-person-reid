from __future__ import division, print_function, absolute_import
import os.path as osp
import warnings
import json, os
from torchreid.data import ImageDataset
from .dataset_processors import processsors as dataset_processors


class ChunkLoader(ImageDataset):
    """Chunk Loader"""

    _junk_pids = [0, -1]
    # @TODO: Need updation once chunks are moved to a dedicated directory
    dataset_dir = "market1501-test"

    def __init__(self, root="", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # @TODO: Need updation once chunks are moved to a dedicated directory
        if osp.isdir(self.dataset_dir):
            data_dir = osp.join(self.dataset_dir, "Market-1501-v15.09.15")
            self.data_dir = data_dir
        else:
            os.makedirs(self.dataset_dir)
            data_dir = osp.join(self.dataset_dir, "Market-1501-v15.09.15")
            self.data_dir = data_dir
            warnings.warn("Chunk not found.")

        self.train_dir = osp.join(self.data_dir, "train_chunks", "c4")
        self.query_dir = osp.join(self.data_dir, "query_sets", "s1")
        self.gallery_dir = osp.join(self.data_dir, "test_sets", "s1")

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]

        self.check_before_run(required_files)

        train = self.load_datasets(self.train_dir)
        query = self.load_datasets(self.query_dir)
        gallery = self.load_datasets(self.gallery_dir)

        print("train ===============")
        print(train)
        print("=====================")

        super(ChunkLoader, self).__init__(train, query, gallery, **kwargs)

    def load_datasets(self, dir_path):
        data_json_path = os.path.join(dir_path, "data.json")
        data = []

        if os.path.exists(data_json_path) == False:
            return []

        data_json = {}
        with open(data_json_path, "r") as json_file:
            data_json = json.load(json_file)

        # use __custom_loader to append json elements into data List
        for key, images in data_json.items():
            image_paths = [os.path.join(dir_path, image) for image in images]
            data.extend(dataset_processors[key](image_paths))

        return data
