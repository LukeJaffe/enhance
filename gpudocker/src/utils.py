import os
import shutil
from distutils.dir_util import copy_tree

# Get docker volume mount string using local and target path info
def get_mount(local_paths, target_dir):
    vol_string = ""
    for local_path in local_paths:
        basename = os.path.basename(local_path)
        target_path = os.path.join(target_dir, basename)
        vol_string += "-v {}:{} ".format(local_path, target_path)
    return vol_string

# Copy libs using local and target path info
def copy_files(local_paths, target_dir):
    for local_path in local_paths:
        basename = os.path.basename(local_path)
        target_path = os.path.join(target_dir, basename)
        # Copy from local to target
        if os.path.isfile(local_path):
            shutil.copy(local_path, target_path) 
        elif os.path.isdir(local_path):
            copy_tree(local_path, target_path) 

def check_abs(file_path):
    if os.path.isabs(file_path):
        return file_path
    else:
        return os.path.abspath(file_path)

