# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import io
import os
import sys
import math
from src.utils import bool_flag
import argparse

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="data generator")

    # main parameters
    parser.add_argument("--data_path", type=str,
                        help="path to input file")
    parser.add_argument("--no_test", type=bool_flag, default=False,
                        help="separate test set, or not?")
    parser.add_argument("--valid_set_size", type=int,
                        help="size of validation set")
    return parser


if __name__ == '__main__':

    params = get_parser().parse_args()
    no_test = params.no_test
    data_path = params.data_path
    trn_path = data_path + '.train'
    vld_path = data_path + '.valid'
    if not no_test:
        tst_path = data_path + '.test'
    vld_tst_size = int(params.valid_set_size)
    assert not os.path.isfile(trn_path)
    assert not os.path.isfile(vld_path)
    if not no_test:
        assert not os.path.isfile(tst_path)
    assert vld_tst_size > 0

    print(f"Reading data from {data_path} ...")
    with io.open(data_path, mode='r', encoding='utf-8') as f:
        lines = [line for line in f]
    total_size = len(lines)
    print(f"Read {total_size} lines.")
    nb_subsets = 1 if no_test else 2
    assert nb_subsets * vld_tst_size < total_size

    alpha = math.log(total_size - 0.5) / math.log(nb_subsets * vld_tst_size)
    assert int((nb_subsets * vld_tst_size) ** alpha) == total_size - 1
    vld_tst_indices = [int(i ** alpha) for i in range(1, nb_subsets * vld_tst_size + 1)]
    if no_test:
        vld_indices = set(vld_tst_indices)
    else:    
        vld_indices = set(vld_tst_indices[::2])
        tst_indices = set(vld_tst_indices[1::2])
    assert len(vld_tst_indices) == nb_subsets * vld_tst_size
    assert max(vld_tst_indices) == total_size - 1
    assert len(vld_indices) == vld_tst_size
    if not no_test:
        assert len(tst_indices) == vld_tst_size

    # sanity check
    total = 0
    power = 0
    while True:
        a = 10 ** power
        b = 10 * a
        s = len([True for x in vld_tst_indices if a <= x < b and x <= total_size])
        if s == 0:
            break
        print("[%12i %12i[: %i" % (a, b, s))
        total += s
        power += 1
    assert total == nb_subsets * vld_tst_size

    print(f"Writing train data to {trn_path} ...")
    print(f"Writing valid data to {vld_path} ...")
    if not no_test:
        print(f"Writing test data to {tst_path} ...")
    f_train = io.open(trn_path, mode='w', encoding='utf-8')
    f_valid = io.open(vld_path, mode='w', encoding='utf-8')
    if not no_test:
        f_test = io.open(tst_path, mode='w', encoding='utf-8')

    for i, line in enumerate(lines):
        if i in vld_indices:
            f_valid.write(line)
        elif not no_test and i in tst_indices:
            f_test.write(line)
        else:
            f_train.write(line)
        if i % 1000000 == 0:
            print(i, end='...', flush=True)

    f_train.close()
    f_valid.close()
    if not no_test:
        f_test.close()
