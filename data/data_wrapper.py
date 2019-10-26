"""
Authors : Kumarage Tharindu & Fan Lei
Class : CSE 575
Organization : ASU CIDSE
Project : SML Project 2
Task : File reader : Provider other package the access to read the data

"""

import os
import scipy.io


def read_data():  # read the data from the Mat

    rel_path = get_data_repo_path()

    data = scipy.io.loadmat(rel_path)

    return data['AllSamples']


def get_data_repo_path():  # get the relative path
    code_dir = os.path.dirname(__file__)  # absolute dir

    rel_path = os.path.join(code_dir, "AllSamples.mat")

    return rel_path






