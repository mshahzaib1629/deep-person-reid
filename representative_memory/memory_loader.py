from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from torchreid.data import ImageDataset


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
            warnings.warn("Representative Memory not found.")

        self.train_dir = self.data_dir

        required_files = [
            self.data_dir,
            self.train_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query =[]
        gallery = []

        print('train len: ', len(train))
        # query.append(train[0])
        # gallery.append(train[0])

        super(RepresentativeMemory, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c(\d)")

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
