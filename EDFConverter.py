#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import subprocess
import warnings


def convert_batch(parentDir, subnames, exePath):
    """
    This function converts a batch of edf files to ascii files. Expects a folder strcuture where
    the parent folder contains subject folders (indicated by subnames) and within each subject
    folder there are edf files. The function will convert all edf files in a folder to ascii
    :param parentDir:
    :param subnames:
    :param exePath
    :return:
    """

    # list subject folders
    subfolders = [fldr for fldr in os.listdir(parentDir) if (os.path.isdir(parentDir + os.sep + fldr)
                                                             and (fldr in subnames))]

    # loop through each subject folder
    for subfldr in subfolders:
        # list the edf files
        edfs = [fl for fl in os.listdir(parentDir + os.sep + subfldr)
                if (os.path.isfile(parentDir + os.sep + subfldr + os.sep + fl) and fl.endswith(".edf"))]

        print("Working on subject: " + subfldr + ". Detected %d files:" % len(edfs))
        print(edfs)

        # loop through the edfs
        for edf in edfs:
            print("Converting " + edf + "...")
            # convert
            convert_file(parentDir + os.sep + subfldr, edf, exePath)


def convert_file(path, name, exePath):
    """
    This function converts an edf file to an ascii file
    :param path:
    :param name:
    :param exePath:
    :return:
    """
    cmd = exePath + os.sep + "edf2asc.exe "
    ascfile = name[:-3] + "asc"

    # check if an asc file already exists
    if not os.path.isfile(path + os.sep + ascfile):
        subprocess.run([cmd, "-p", path, path + os.sep + name])
    else:
        warnings.warn("An Ascii file for " + name + " already exists!")


def main():
    dataDir = r"C:\Users\abdou\Downloads\Blumenfeld Lab\TWCF\Eye Tracking\Data\Exp 2"
    subnames = ["TA272"]
    exeDir = r"C:\Users\abdou\Downloads\Blumenfeld Lab\TWCF\Eye Tracking\Python_Code"

    convert_batch(dataDir, subnames, exeDir)


if __name__ == "__main__":
    main()

