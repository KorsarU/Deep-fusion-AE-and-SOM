import os
from os import walk

#Fetches array of filenames for a given folder path
def test_frame_iterator(path):
    file_names = []
    for (dirpath, dirnames, filenames) in walk(path):
        file_names.extend(filenames)
        break
    return file_names

#Gets all branches for a given folder
def get_unique_paths(path):
    frames_paths = []
    unique_paths = list()
    for subdir, dirs, files in os.walk(path):
        for file in files:
            filepath = subdir + os.sep
            frames_paths.append(filepath)
            unique_paths = list(set(frames_paths))
    return unique_paths




