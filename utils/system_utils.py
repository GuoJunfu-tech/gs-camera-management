#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os
import re


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def parse_dimensions(dim_str: str):
    """input value like "200x300", output [200,300]

    Args:
        dim_str (str): strings like "mxn"


    Returns:
        list(int): [m,n]
    """
    match = re.match(r"^(\d+)x(\d+)$", dim_str)
    if match:
        m, n = map(int, match.groups())
        return [m, n]
    else:
        raise ValueError("the form of input should be 'nxm'，e.g. '300x200'")
