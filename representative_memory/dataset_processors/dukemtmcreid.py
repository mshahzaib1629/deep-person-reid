import re
def dukemtmcreid_processor(img_paths, relabel=True):
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    data = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        assert 1 <= camid <= 8
        camid -= 1 # index starts from 0
        if relabel:
            pid = pid2label[pid]
        data.append((img_path, pid, camid))

    return data