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
from sympy import factorint
import collections
import argparse
import re
import numpy as np
from src.utils import bool_flag, initialize_exp
triple_list = ['aaa', 'bbb', 'ccc', 'aab', 'bbc', 'cca', 'aac', 'bba', 'ccb', 'aae', 'bbf', 'ccd', 'aaf', 'bbd', 'cce',
               'afa', 'bdb', 'cec', 'aff', 'bdd', 'cee']
quad_list = ['dddd', 'bbbd', 'bdbd', 'bbdd', 'dbdd', 'fbdd', 'dbbd', 'cddd']

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="data generator")

    # main parameters
    parser.add_argument("--input_dir", type=str, default="",
                        help="directory where input file lives")
    parser.add_argument("--output_file", type=str, default="",
                        help="path to output file")
    parser.add_argument("--loops", type=int,
                        help="number of loops to generate data at")
    parser.add_argument("--distance", type=int,
                        help="max strikeout distance", default=0)
    parser.add_argument("--in_modulo", type=int,
                        help="modulo for the input coefs", default=0)
    parser.add_argument("--is_aef", choices=[None, "ae", "aef"], default=None)
    parser.add_argument("--out_modulo", type=int,
                        help="modulo for the output coefs", default=0)
    #shouldn't need to do this here, can just read/write raw and decode inside the dataset object
    parser.add_argument("--prime_encoding", choices = [None,"coef","parents","both"], default=None)
    parser.add_argument("--base", type=int, default=1000)
    parser.add_argument("--case", choices = [None,"sorted","skip_odd","skip_even","compress","zero_or_not","zero_and_sign","zero_and_sign_sorted","mag_only","dup_pattern","multiplicity"], default=None)
    parser.add_argument("--unique", type=bool_flag, help="only keep unique x,y pairs", default=False)
    parser.add_argument("--sign_opt", choices = [None,"only","last"], default=None)
    parser.add_argument("--orbit", choices=["", "single_quad", "single_triple", "single_quadtrip"], default="")
    return parser

def readESymb(loop, path, file, quad, ae, aef, octuples=False):
    name = os.path.join(path, file)
    assert os.path.isfile(name)
    res = ''
    if octuples:
        prefix = 'Esymboct'
    elif quad:
        prefix = 'Esymbquad'
    elif ae:
        prefix = 'Eae'
    elif aef:
        prefix = 'Eaef'
    else:
        prefix = 'Esymb'
    with open(name, 'rt') as f:
        reading_form = False
        for line in f:
            if not reading_form:
                if not line.startswith(prefix + '[' + str(loop) + ']'): continue
                res = ''
                reading_form = True
            res += line[:-2] if line[-2] == '\\' else line[:-1]
            if line[-2] in [":", ";"]:
                break
    return res


def convert(loop, path, file, quad,ae,aef,octuples=False, orbit=""):
    if octuples:
        base = readESymb(loop, path, file, False, ae, aef, True)[:-2]
        base = re.sub(' ', '', base)
        print(base[-10:])
        t = re.split(":=\[|\),|\)\]", base)[1:]
        if len(t[-1]) == 0: t = t[:-1]
        print(len(t), "elements")
        s = [re.split(":=|SB\(|\)", re.sub('[, *]', '', tt)) for tt in t]
        prefix = [f"v{i:02d}" for i in range(93)]
        dev = []
        for i, ss in enumerate(s):
            for j, tt in enumerate(ss[1::2]):
                s[i][1 + 2 * j] = prefix[i] + tt
            dev += s[i]
        print(len(dev), "terms")
        if len(dev[-1]) == 0: dev = dev[:-1]
        print(dev[-10:])
    elif quad:
        base = readESymb(loop, path, file, True, ae, aef)[:-2]
        base = re.sub(' ', '', base)
        print(base[-10:])
        t = re.split(":=\[|\),|\)\]", base)[1:]
        if len(t[-1]) == 0: t = t[:-1]
        print(len(t), "elements")
        s = [re.split(":=|SB\(|\)", re.sub('[, *]', '', tt)) for tt in t]
        prefix = list("abcdefgh")
        dev = []
        for i, ss in enumerate(s):
            for j, tt in enumerate(ss[1::2]):
                s[i][1 + 2 * j] = prefix[i] + tt
            dev += s[i]
        print(len(dev), "terms")
        if len(dev[-1]) == 0: dev = dev[:-1]
        print(dev[-10:])
    else:
        dev = re.split(":=|SB\(|\)", re.sub('[,*]', '', readESymb(loop, path, file, False, ae, aef)))[1:-1]
    keys = dev[1::2]
    values = [int(re.sub('[+-]$', t[0] + '1', t)) for t in dev[0::2]]
    out_dict = {}
    for k, v in zip(keys, values):
        if orbit == "single_quad":
            # only get one quad
            for elem in quad_list:
                if k.endswith(elem):
                    out_dict[k] = v
        elif orbit == "single_triple":
            # only get one triple
            for elem in triple_list:
                if k.startswith(elem):
                    out_dict[k] = v
        elif orbit == "single_quadtrip":
            # only get one quad and one triple
            for elem in triple_list:
                for elem2 in quad_list:
                    if k.startswith(elem) and k.endswith(elem2):
                        out_dict[k] = v
        else:
            out_dict[k] = v
    return out_dict


def encode(number, base, mod=0):
    if mod != 0:
        number = number % mod
    if base <= 1:
        return [str(number)]
    if number != 0:
        prefix2 = []
        w = abs(number)
        while w > 0:
            prefix2.append(str(w % base))
            w = w // base
        prefix2 = prefix2[::-1]
    else:
        prefix2 = ['0']
    prefix2 = (['+'] if number >= 0 else ['-']) + prefix2
    return prefix2

def encode_primes(value,base=1000):
    prefix = []
    if abs(value) == 1:
        factors = {1: 1}
    else:
        factors = factorint(abs(value))
    remainder = 0
    for fac in factors.keys():
        if fac < 11 and fac > 0:
            remainder = value / pow(fac, factors[fac])
            prefix += ([f"{fac}"] + [f"E{factors[fac]}"])
        if remainder != 0:
            prefix = []
            w = abs(remainder)
            while w > 0:
                prefix.append(str(w % base))
                w = w // base
            prefix = prefix[::-1]
        else:
            prefix = ['0']
    prefix = (['+'] if value >= 0 else ['-']) + prefix
    return prefix

def encode_mag(number, base, mod=0):
    if mod != 0:
        number = number % mod
    if base <= 1:
        return [str(number)]
    if number != 0:
        prefix2 = []
        w = abs(number)
        while w > 0:
            prefix2.append(str(w % base))
            w = w // base
        prefix2 = prefix2[::-1]
    else:
        prefix2 = ['0']
    prefix2 = ['+'] + prefix2
    return prefix2


def encode_zeros(number, do_sign=False):
    if number != 0:
        prefix2 = ['1']
    else:
        prefix2 = ['0']
    if do_sign:
        prefix2 = (['+'] if number >= 0 else ['-']) + prefix2
    return prefix2


def get_loop_data(loop, path, ae, aef, orbit):
    if ae: return convert(loop, path, "Eae", False, ae, False,False, orbit)
    elif aef: return convert(loop, path, "Eaef", False, False, aef,False, orbit)
    else:
        if loop < 6:
            return convert(loop, path, "EZ_symb_new_norm", False, False, False,False, orbit)
        if loop == 6:
            return convert(loop, path, "EZ6_symb_new_norm", False, False, False, False, orbit)
    print("loop up to 6")
    return None


def query_loop(key, data):
    if key in data.keys():
        return data[key]
    return 0


def make_lists(params):
    path = params.input_dir
    loop = params.loops
    if loop > 6:
        print("error! strikeout works on canonical rep, > 6 loop data is in quad rep")
    distance = params.distance
    case=params.case
    prime_encoding = params.prime_encoding
    base = params.base
    mod=params.in_modulo
    outmod=params.out_modulo
    orbit=params.orbit
    ae = (params.is_aef == 'ae')
    aef = (params.is_aef == 'aef')
    unique = params.unique
    if case == "skip_odd": skip = 1
    elif case == "skip_even": skip = 2
    else: skip = 0

    e3 = get_loop_data(loop - 1, path, ae, aef, orbit)
    e4 = get_loop_data(loop, path, ae, aef, orbit)
    vals = set()
    input = []
    output = []
    maxdist = 1000 if distance == 0 else distance

    if skip > 0: incr = 2
    else: incr = 1
    if skip < 2: beg = 0
    else: beg = 1

    for key, value in e4.items():
        nl = len(key)
        lst = []
        for j in range(beg, nl - 1, incr):
            for k in range(j + 1, min(j + maxdist + 1, nl)):
                key2 = key[:j] + key[j + 1:k] + key[k + 1:]
                if orbit == "single_quad":
                    # only get one quad
                    for elem in quad_list:
                        if key2.endswith(elem):
                            lst.append(query_loop(key2, e3))
                elif orbit == "single_triple":
                    # only get one triple
                    for elem in triple_list:
                        if key2.startswith(elem):
                            lst.append(query_loop(key2, e3))
                elif orbit == "single_quadtrip":
                    # only get one quad and one triple
                    for elem in triple_list:
                        for elem2 in quad_list:
                            if key2.startswith(elem) and key2.endswith(elem2):
                                lst.append(query_loop(key2, e3))
                else:
                    lst.append(query_loop(key2, e3))

        if case=="sorted":
            lst.sort()
        if case=="compress":
            #note that this ONLY compresses original parents- if they're identical mod something,
            # the mod arg will not compress them. Could add this feature, but maybe not interesting
            lst = [*set(lst)]
            lst.sort()
        if case=="dup_pattern":
            seen = []
            pattern = []
            for w in lst:
                if not w in seen:
                    seen.append(w)
                    pattern.append(len(seen) - 1)
                else:
                    pattern.append(seen.index(w))
            lst = pattern
        flst = []
        if case == "multiplicity":
            counter = collections.Counter(lst)
            for (elem, count) in counter.items():
                flst.extend(encode(elem, base, mod))
                flst.extend("(")
                flst.extend(encode(count, 0, 0))
                flst.extend(")")
        else:
            for w in lst:
                if case=="zero_or_not":
                    flst.extend(encode_zeros(w, False))
                elif case=="zero_and_sign":
                    flst.extend(encode_zeros(w, True))
                elif case=="zero_and_sign_sorted":
                    flst.extend(encode_zeros(w, True))
                    lst.sort()
                elif case=="mag_only":
                    flst.extend(encode_mag(w, base, mod))
                elif prime_encoding=="parents" or prime_encoding=="both":
                    flst.extend(encode_primes(w))
                else:
                    flst.extend(encode(w, base, mod))

        if prime_encoding == "coef" or prime_encoding == "both":
            olst=encode_primes(value)
        else: # what if we don't mod out the target?
            olst = encode(value, base, outmod)

        #unique? do for unique, compress
        if unique:
            # print(flst+olst)
            if tuple(flst + olst) in vals:
                continue
            vals.add(tuple(flst + olst))

        input.append(flst)
        output.append(olst)
    return input, output


def export_hash(data, outfile, binary, sign_only, sign_last, base, octuples=False):
    file_handler = io.open(outfile, mode="wt", encoding="utf-8")
    for i, (k, v) in enumerate(data.items()):
        if octuples:
            prefix1_str = k[:3] + " " + " ".join(k[3:])
        else:
            prefix1_str = " ".join(k)
        if binary:
            file_handler.write(f"{i + 1}|{prefix1_str}\t1\n")
        else:
            prefix2 = encode(v, base)
            if sign_last:
                prefix2 = prefix2[1:] + prefix2[:1]
            if sign_only:
                prefix2 = prefix2[:1]
            prefix2_str = " ".join(prefix2)
            file_handler.write(f"{i + 1}|{prefix1_str}\t{prefix2_str}\n")
        file_handler.flush()
    file_handler.close()


def export_pairs(idata, odata, outfile):
    file_handler = io.open(outfile, mode="wt", encoding="utf-8")
    for i, (k, v) in enumerate(zip(idata, odata)):
        prefix1_str = " ".join(k)
        prefix2_str = " ".join(v)
        file_handler.write(f"{i + 1}|{prefix1_str}\t{prefix2_str}\n")
        file_handler.flush()
    file_handler.close()

if __name__ == '__main__':

    params = get_parser().parse_args()
    print(f"Reading data from {params.input_dir} ...")
    i, o = make_lists(params)
    print(f"Exporting to {params.output_file}.data ...")
    export_pairs(i, o, params.output_file + ".data")

    print(f"Done ... ")
    print(f"cat {params.output_file}.data | shuf > {params.output_file}.prefix")
    print(f"To create final file ... ")