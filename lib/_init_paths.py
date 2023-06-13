import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


root_dir = osp.abspath(osp.dirname(osp.join(__file__, '..')))

lib_path = osp.join(root_dir, 'lib')
add_path(lib_path)

