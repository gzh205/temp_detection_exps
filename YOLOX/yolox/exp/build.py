#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import importlib
import os
import platform
import sys


def get_exp_by_file(exp_file):
    #try:
    sys.path.append(os.path.dirname(exp_file))
    current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
    exp = current_exp.Exp()
    return exp
    #except Exception:
    #    raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))


'''
elif sys == 'Linux':
filename = os.path.splitext(exp_file)[0]
        filename = os.path.splitdrive(filename)[1]
        module_name = ''
        while True:
            filename, tmp_name = os.path.split(filename)
            if filename == "" or filename == ".":
                module_name = tmp_name + module_name
                break
            module_name = '.' + tmp_name + module_name
        exp = eval(module_name + '.Exp()')
'''


def get_exp_by_name(exp_name):
    exp = exp_name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"
    module_name = ".".join(["yolox", "exp", "default", exp])
    exp_object = importlib.import_module(module_name).Exp()
    return exp_object


def get_exp(exp_file=None, exp_name=None):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    We add a platform recognition in this function.
    if linux, we use default load exp code.
    if windows, we use pyinstaller to load exp code.
    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
            exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if sys.platform.__contains__("win"):
        project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        exp_file = os.path.join(project_path, exp_file)
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    else:
        return get_exp_by_name(exp_name)
