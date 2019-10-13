# Filerenamer
As the name suggests is a python module which helps you rename collection of files in a folder.

# Capabilities

Currently in its first draft, the filerenammer is able to do following.

1- Rename all files in a folder
2- Append a prefix to all file names.
3- Append a suffix to all file names.
4- Append random integers uniformly distributed between a min and max given range

# Usage

To rename files

Before:
myfolder/myfile.txt

Run Command:
renamer.py -d myfolder -p myprefix -s mysuffix -m 1 -n 100

After:
myfolder/myprefix_myfile_55_mysuffix.txt


