from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings
import json, os
from torchreid.data import ImageDataset


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
            self.data_dir = self.dataset_dir
        else:
            os.makedirs(self.dataset_dir)
            self.data_dir = self.dataset_dir
            warnings.warn("Representative Memory not found.")

        self.train_dir = self.data_dir

        required_files = [
            self.data_dir,
            self.train_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = []
        gallery = []

        # print('train [0]: ', train[0])
        # print('train [1]: ', train[1])

        super(RepresentativeMemory, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c(\d)")

        labels_dict = get_image_label_dict(dir_path)

        pid_container = set()
        for img_path in img_paths:
            # pid, _ = map(int, pattern.search(img_path).groups())
            file_name = img_path.split("/")[-1]
            pid = int(labels_dict[file_name])

            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # print("pid2label: ", pid2label)
        data = []
        for img_path in img_paths:
            # pid, camid = map(int, pattern.search(img_path).groups())
            file_name = img_path.split("/")[-1]
            pid = int(labels_dict[file_name])
            if pid == -1:
                continue  # junk images are just ignored
            # @TODO: Need to discuss if we need camera info
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            # camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            print(file_name, " => ", pid)
            data.append((img_path, pid, 0))

        return data
