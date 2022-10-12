''' Tools for loading the dataset
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from __future__ import annotations
from typing import Iterable
from os import PathLike

# tpl imports
from alive_progress import alive_it


def get_source_filenames(root: PathLike, extensions: Iterable[str] = ['C', 'cc', 'cxx', 'cpp', 'c', 'h', 'hpp'], show_progress: bool = True) -> list[PathLike]:
    ''' return a list of all the filenames of source files with the given extensions in root.
        Args:
            root: where to start searching for files. Is searched recursively.
            extensions: what extensions define the source files. C/C++ extensions by default.
            show_progress: If true, then display a progress bar.

        Returns:
            A list of paths to all the source files.
    '''
    from glob import glob
    from os.path import join as path_join, isdir, exists, islink

    def is_valid_source_file(fname: PathLike) -> bool:
        return (not isdir(fname)) and exists(fname)
    
    all_files = []
    vals = alive_it(extensions, title='Searching for source files'.rjust(26)) if show_progress else extensions
    for ext in vals:
        files = glob(path_join(root, '**', '*.' + ext), recursive=True)
        all_files.extend( [f for f in files if is_valid_source_file(f)] )

    return all_files


def filter_bad_encoding(fnames: Iterable[PathLike], show_progress: bool = True) -> list[PathLike]:
    ''' Remove files with non utf-8 encodings.
        Args:
            fnames: a list of filenames to filter.
            show_progress: If true, then display a progress bar.

        Returns:
            A copy of fnames with files that contained non-utf-8 characters filtered out.
    '''
    results = []
    vals = alive_it(fnames, title='Removing non-utf-8'.rjust(26)) if show_progress else fnames
    for f in vals:
        try:
            for _ in open(f, 'r'):
                pass
            results.append(f)
        except:
            pass
    return results


def get_loc(flist: Iterable[PathLike], show_progress: bool = True) -> int:
    ''' Returns the total number of lines in all the files in flist.
        Args:
            flist: a list of filenames to count LOC in.
            show_progress: If true, then display a progress bar.
        
        Returns:
            The total LOC summed over all the files.
    '''
    #import subprocess
    LOC = 0
    vals = alive_it(flist, title='Counting LOC'.rjust(26)) if show_progress else flist
    for fname in vals:
        #LOC += int(subprocess.check_output(['wc', '-l', fname]).split()[0])
        LOC += sum(1 for _ in open(fname, 'r', errors='ignore'))
    return LOC


def get_loc_per_extension(flist: Iterable[PathLike], show_progress: bool = True) -> int:
    ''' Returns the total number of lines in all the files in flist per extension.
        Args:
            flist: a list of filenames to count LOC in.
            show_progress: If true, then display a progress bar.
        
        Returns:
            The total LOC summed over all the files stored in buckets in a dict.
    '''
    from os.path import splitext
    get_extension = lambda x: splitext(x)[-1]

    LOC = {}
    vals = alive_it(flist, title='Counting LOC'.rjust(26)) if show_progress else flist
    for fname in vals:
        ext = get_extension(fname)
        if ext not in LOC:
            LOC[ext] = 0

        LOC[ext] += sum(1 for _ in open(fname, 'r', errors='ignore'))
        
    return LOC


def get_source_file_size(flist: Iterable[PathLike], show_progress: bool = True) -> int:
    ''' Return the data set size based on a list of fnames in bytes.
        Args:
            flist: a list of filenames to sum the size over.
            show_progress: If true, then display a progress bar.

        Returns:
            The total number of bytes that flist files takes up.
    '''
    from os.path import getsize

    num_bytes = 0
    vals = alive_it(flist, title='Calculating dataset size'.rjust(26)) if show_progress else flist
    for fname in vals:
        num_bytes += getsize(fname)
    return num_bytes

