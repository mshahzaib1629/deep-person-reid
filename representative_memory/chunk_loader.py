from __future__ import division, print_function, absolute_import
import os.path as osp
import warnings
import json, os
from torchreid.data import ImageDataset
from .utils import process_datasets


class ChunkLoader(ImageDataset):
    """Chunk Loader"""

    TRAIN_CHUNK = "c14"
    QUERY_CHUNK = "c11"
    GALLERY_CHUNK = "c11"

    _junk_pids = [0, -1]
    dataset_dir = "chunks"

    def __init__(self, root="", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        if osp.isdir(self.dataset_dir):
            self.data_dir = self.dataset_dir
        else:
            os.makedirs(self.dataset_dir)
            self.data_dir = self.dataset_dir
            warnings.warn("Chunk not found.")

        self.check_directories()
        self.train_dir = osp.join(self.data_dir, "train", self.TRAIN_CHUNK)
        self.query_dir = osp.join(self.data_dir, "query", self.QUERY_CHUNK)
        self.gallery_dir = osp.join(self.data_dir, "gallery", self.GALLERY_CHUNK)

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]

        self.check_before_run(required_files)

        train = process_datasets(self.train_dir)
        query = process_datasets(self.query_dir)
        gallery = process_datasets(self.gallery_dir)

        super(ChunkLoader, self).__init__(train, query, gallery, **kwargs)

    def check_directories(self):
        """This function will check if required folder structure is available or not. If not, it will create base directories for train, query and gallery chunks."""

        train_dir = osp.join(self.data_dir, "train")
        if osp.exists(train_dir) == False:
            os.makedirs(train_dir)

        query_dir = osp.join(self.data_dir, "query")
        if osp.exists(query_dir) == False:
            os.makedirs(query_dir)

        gallery_dir = osp.join(self.data_dir, "gallery")
        if osp.exists(gallery_dir) == False:
            os.makedirs(gallery_dir)
