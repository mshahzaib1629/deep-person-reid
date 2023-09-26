from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings
import json, os
from torchreid.data import ImageDataset
from .utils import process_datasets


def get_image_label_dict(representative_memory_directory):
    labels_file_path = os.path.join(representative_memory_directory, "labels.json")

    label_json_data = {}
    if os.path.exists(labels_file_path):
        with open(labels_file_path, "r") as json_file:
            label_json_data = json.load(json_file)
            json_file.close()

    return label_json_data


class RepresentativeMemory(ImageDataset):
    """Representative Memory"""

    _junk_pids = [0, -1]
    dataset_dir = "representative-memory"

    def __init__(self, root="", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        if osp.isdir(self.dataset_dir):
            self.memory_dir = os.path.join(self.dataset_dir, "memory")
        else:
            os.makedirs(self.dataset_dir)
            self.memory_dir = os.path.join(self.dataset_dir, "memory")
            warnings.warn("Representative Memory not found.")

        if os.path.exists(self.memory_dir) == False:
            os.makedirs(self.memory_dir)

        required_files = [
            self.memory_dir,
        ]
        self.check_before_run(required_files)

        train = process_datasets(self.memory_dir)
        query = []
        gallery = []

        super(RepresentativeMemory, self).__init__(train, query, gallery, **kwargs)
