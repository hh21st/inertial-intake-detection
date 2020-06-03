import os
import shutil
import logging
from pathlib import Path
import ntpath
import glob

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO)

def get_bool(boolean_string):
    if boolean_string not in {'False', 'True'}:
        raise ValueError('{0} is not a valid boolean string'.format(boolean_string))
    return boolean_string == 'True'

def create_dir_if_required(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def copy_file(full_file_name, copy_to_dir):
    create_dir_if_required(copy_to_dir)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, copy_to_dir)    
    else:
        logging.error("file {0} does not exist.".format(full_file_name))

def move_file(file_name, soure, destination):
    source_file_fullname=os.path.join(soure,file_name)
    destination_file_fullname=os.path.join(destination, file_name)
    if os.path.isfile(source_file_fullname):
        if os.path.isfile(destination_file_fullname):
            os.remove(destination_file_fullname)
        shutil.move(source_file_fullname, destination_file_fullname)
    else:
        logging.error("file {0} does not exist.".format(source_file_fullname))

def get_immediate_subdirnames(root_dir):
    return [name for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))]

def get_immediate_subdirs(root_dir):
    return [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))]

def get_immediate_files(root_dir):
    return [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, name))]

def get_parent_dir(path):
    path_finder = Path(path)
    return path_finder.parent

def get_current_dir_name(path):
    return os.path.basename(path)

def get_file_name_from_path(path, exclude_extension = False):
    head, tail = ntpath.split(path)
    filename = tail or ntpath.basename(head)
    if exclude_extension:
        filename = get_file_name_without_extension(filename)
    return filename

def is_file(file_path):
    return True if os.path.isfile(file_path) else False

def delete_file_if_exists(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)

def get_file_extension(file_name):
    return os.path.splitext(file_name)[1]

def get_file_name_without_extension(file_name):
    return os.path.splitext(file_name)[0]

def calc_recall(tp, fn, rounding_digits = None):
    recall = tp / (tp + fn)
    if rounding_digits != None:
        recall = round(recall, rounding_digits)
    return recall

def calc_precision(tp, fp, rounding_digits = None):
    precision = tp / (tp + fp)
    if rounding_digits != None:
        precision = round(precision, rounding_digits)
    return precision

def count_file_lines(filename, exclude_first_line = False):
    with open(filename) as f:
        count = sum(1 for line in f)
    if exclude_first_line:
        count -=1
    return count

def get_files_in_dir(dir, extensionPattern = None):
    if(extensionPattern==None):
        return [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    else:
        return glob.glob(os.path.join(dir, extensionPattern))

def count_files_lines_in_dir(dir, extensionPattern = None, exclude_first_line = False):
    count = 0
    filenames = get_files_in_dir(dir, extensionPattern)
    for filename in filenames:
        count += count_file_lines(filename, exclude_first_line)
    return count



if __name__ == '__main__':
    #test:
    result=count_files_lines_in_dir(r'C:\H\PhD\ORIBA\Model\FileGen\OREBA.dis\64_std_uni_no_smo', '*.csv', True)
    print(result)
