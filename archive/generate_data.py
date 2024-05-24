# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import io
import os
import sys
import math
import re
import numpy as np
import argparse
from sympy import factorint
from rels_utils import output_rel_instances_jsons, rels_to_generate, rels_to_generate_compact_default, generate_trivial0_symb
from src.utils import bool_flag

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
    parser.add_argument("--input_file", type=str, default=None,
                        help="full path to input file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="full path to output file")
    parser.add_argument("--rels_output_path", type=str, default=None,
                        help="path to output relations")
    parser.add_argument("--loops", type=int,
                        help="number of loops to generate data at")
    parser.add_argument("--base", type=int, default=1000)
    parser.add_argument("--case", choices=[None, "b", "quad", "octuples", "zero", "quad_zero"], default=None)
    parser.add_argument("--trivial_frac", type=float, default=0.05)
    parser.add_argument("--is_aef", choices=[None, "ae", "aef"], default=None)
    parser.add_argument("--prime_encoding", type=bool_flag, default=False)
    parser.add_argument("--sign_opt", choices=[None, "sign_only", "last", "mag_only"], default=None)
    parser.add_argument("--orbit", choices=["", "single_quad", "single_triple", "single_quadtrip"], default="")
    parser.add_argument("--generate_relations", type=bool_flag, default=False,
                        help="Generate relations.")
    parser.add_argument("--raw_symbol_json", type=bool_flag, default=False, help="write the nonzero elems of symbol as a json")

    return parser

def readESymb(loop, name, quad, ae, aef, octuples=False):
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

def convert(loop, filename, quad, ae, aef, octuples=False, orbit=""):
    if octuples:
        base = readESymb(loop, filename, False, ae, aef, True)[:-2]
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
        base = readESymb(loop, filename, True, ae, aef)[:-2]
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
        dev = re.split(":=|SB\(|\)", re.sub('[,*]', '', readESymb(loop, filename, False, ae, aef)))[1:-1]
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

def encode_primes(value,base=1000):
    prefix = []
    if abs(value) == 1:
        factors = {1: 1}
    else:
        factors = factorint(abs(value))
    remainder = value
    for fac in factors.keys():
        if fac < 11 and fac > 0:
            remainder = int(remainder / pow(fac, factors[fac]))
            prefix += ([f"{int(fac)}"] + [f"E{factors[fac]}"])
    if remainder != 0:
        suffix = []
        w = abs(remainder)
        if w != 1:
            while w > 0:
                suffix.append(str(w % base))
                w = w // base
            prefix += suffix[::-1]
    else:
        prefix = ['0']
    prefix = (['+'] if value >= 0 else ['-']) + prefix
    return prefix

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
def encode_sign(number):
    if number != 0:
        prefix2 = ['1']
    else:
        prefix2 = ['0']
    prefix2 = (['+'] if number >= 0 else ['-']) + prefix2
    return prefix2

def encode_zeros(number, do_sign=False):
    if number != 0:
        prefix2 = ['1']
    else:
        prefix2 = ['0']
    if do_sign:
        prefix2 = (['+'] if number >= 0 else ['-']) + prefix2
    return prefix2


def query_loop(key, data):
    if key in data.keys():
        return data[key]
    return 0


def export_hash(data, outfile, binary, sign_only, sign_last, base, octuples=False, mag_only=False, prime_encoding=False):
    file_handler = io.open(outfile, mode="wt", encoding="utf-8")
    for i, (k, v) in enumerate(data.items()):
        if octuples:
            prefix1_str = k[:3] + " " + " ".join(k[3:])
        else:
            prefix1_str = " ".join(k)
        if binary:
            file_handler.write(f"{i + 1}|{prefix1_str}\t1\n")
        else:
            if prime_encoding:
                prefix2 = encode_primes(v)
            else:
                prefix2 = encode(v, base)
            if sign_last:
                prefix2 = prefix2[1:] + prefix2[:1]
            if sign_only:
                prefix2 = prefix2[:1] + ["1"]
            if mag_only:
                prefix2 = ["+"] + prefix2[1:]
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


# could define a stricter approach (to learn the harder ruels)
def new_key(loops, ae, aef):
    if ae:
        return ''.join(np.random.choice(list("ae"), 2 * loops))
    elif aef:
        return ''.join(np.random.choice(list("aef"), 2 * loops))
    else:
        return ''.join(np.random.choice(list("abcdef"), 2 * loops))
def gen_next(letter):
    #hardcode adjacency rules to generate nontrivial zeroes
    if letter == 'a':
        return np.random.choice(list("abcef"))
    if letter == 'b':
        return np.random.choice(list("abcdf"))
    if letter == 'c':
        return np.random.choice(list("abcde"))
    if letter == 'd':
        return np.random.choice(list("bcd"))
    if letter == 'e':
        return np.random.choice(list("ace"))
    if letter == 'f':
        return np.random.choice(list("abf"))

def allowed_quads_for_stem(stem):
    #hardcode adjacency rules to generate nontrivial zeroes
    if stem[-1] == 'a':
        return [i + stem for i in list("bcdfh")]
    if stem[-1] == 'b':
        return [i + stem for i in list("abcdefgh")]
    if stem[-1] == 'c':
        return [i + stem for i in list("abcdegh")]
    if stem[-1] == 'd':
        return [i + stem for i in list("abcdegh")]
    if stem[-1] == 'e':
        return [i + stem for i in list("h")]
    if stem[-1] == 'f':
        return [i + stem for i in list("bcdf")]

def gen_quad_suffix(letter):
    #hardcode adjacency rules to generate nontrivial zeroes
    if letter == 'a':
        return np.random.choice(list("bcdfh"))
    if letter == 'b':
        return np.random.choice(list("abcdefgh"))
    if letter == 'c':
        return np.random.choice(list("abcdegh"))
    if letter == 'd':
        return np.random.choice(list("abcdegh"))
    if letter == 'e':
        return np.random.choice(list("h"))
    if letter == 'f':
        return np.random.choice(list("bcdf"))

def gen_last(letter):
    #given the second to last letter, generate a valid last letter
    if letter == 'a':
        return np.random.choice(list("ef"))
    if letter == 'b':
        return np.random.choice(list("df"))
    if letter == 'c':
        return np.random.choice(list("de"))
    if letter == 'd':
        return np.random.choice(list("d"))
    if letter == 'e':
        return np.random.choice(list("e"))
    if letter == 'f':
        return np.random.choice(list("f"))

def new_quad_stem(loops):
    # generate a key that is not a trivial zero.
    key = []
    letter = np.random.choice(list("abc"))
    key.append(letter)
    i = 0
    while i < (2 * loops - 5):
        letter = gen_next(letter)
        key.append(letter)
        i += 1
    return ''.join(key)

def new_nontriv_key(loops, format):
        if format == "quad":
        # generate a key that is not a trivial zero.
            key = []
            letter = np.random.choice(list("abc"))
            key.append(letter)
            i = 0
            while i < (2 * loops - 5):
                letter = gen_next(letter)
                key.append(letter)
                i += 1
            # generate the last letter
            letter = gen_quad_suffix(letter)
            key.insert(0, letter)
            i += 1
            return ''.join(key)
        else:
            # generate a key that is not a trivial zero.
            key=[]
            letter = np.random.choice(list("abc"))
            key.append(letter)
            i = 0
            while i < (2*loops-2):
                letter = gen_next(letter)
                key.append(letter)
                i += 1
            # generate the last letter
            letter = gen_last(letter)
            key.append(letter)
            i += 1
            return ''.join(key)

def build_zero_file(number, trivial_frac, data, loops, outfile, binary, sign_last, ae, aef, quad):
    if quad: format = "quad"
    else: format = "full"
    file_handler = io.open(outfile, mode="wt", encoding="utf-8")
    num_triv = int(trivial_frac * number)
    num_per_type_triv = math.floor(num_triv/18)
    num_nontriv=number-18*num_per_type_triv
    trivcounter=0
    keysdict={}
    #if we're doing ae or aef, can't restrict trivial zeroes to a fraction- not enough of them
    if ae or aef:
        for i in range(number):
            # generate a word
            key = new_key(loops, ae, aef)
            # if it's not zero, keep going until it is
            while ((key in data.keys()) or (key in keysdict.keys())):
                key = new_key(loops, ae, aef)
                keysdict[key]=0
            # log it
            prefix1_str = " ".join(key)
            if binary:
                file_handler.write(f"{i + 1}|{prefix1_str}\t0\n")
            elif sign_last:
                file_handler.write(f"{i + 1}|{prefix1_str}\t0 +\n")
            else:
                file_handler.write(f"{i + 1}|{prefix1_str}\t+ 0\n")
            file_handler.flush()
    else:
        if format == "quad":
            i = 0
            #if we do 10k draws and still don't find a new nontrivial zero, we've probably run out
            max_tries = 10000
            stemsdict={}
            #generate nontrivial zeroes
            while i < num_nontriv:
                #get a quad stem we haven't seen before
                stem = new_quad_stem(loops)
                num_tries = 0
                while (stem in stemsdict.keys() and num_tries < max_tries):
                    stem = new_quad_stem(loops)
                    num_tries += 1
                if num_tries == max_tries:
                    #if we are out of nontrivial zeroes, fill the rest with trivials and break the loop
                    num_triv = number-i
                    num_per_type_triv = math.floor(num_triv / 18)
                    break
                stemsdict[stem]=0

                # add all possible quads to it
                allowed_quads = allowed_quads_for_stem(stem)
                for key in allowed_quads:
                    if key in data.keys(): continue
                    #log it
                    prefix1_str = " ".join(key)
                    if binary:
                        file_handler.write(f"{i + 1}|{prefix1_str}\t0\n")
                    elif sign_last:
                        file_handler.write(f"{i + 1}|{prefix1_str}\t0 +\n")
                    else:
                        file_handler.write(f"{i + 1}|{prefix1_str}\t+ 0\n")
                    file_handler.flush()
                    i += 1

        else:
            #generate nontrivial zeroes
            for i in range(num_nontriv):
                key = new_nontriv_key(loops, format)
                while ((key in data.keys()) or (key in keysdict.keys())):
                    key = new_nontriv_key(loops, format)
                keysdict[key]=0
                #log it
                prefix1_str = " ".join(key)
                if binary:
                    file_handler.write(f"{i + 1}|{prefix1_str}\t0\n")
                elif sign_last:
                    file_handler.write(f"{i + 1}|{prefix1_str}\t0 +\n")
                else:
                    file_handler.write(f"{i + 1}|{prefix1_str}\t+ 0\n")
                file_handler.flush()

        # generate trivial zeroes
        trivial_zeroes = generate_trivial0_symb(num_per_type_triv, 6, format=format)
        for i,key in enumerate(trivial_zeroes.keys()):
            keysdict[key]=0
            # log it
            prefix1_str = " ".join(key)
            if binary:
                file_handler.write(f"{i + 1}|{prefix1_str}\t0\n")
            elif sign_last:
                file_handler.write(f"{i + 1}|{prefix1_str}\t0 +\n")
            else:
                file_handler.write(f"{i + 1}|{prefix1_str}\t+ 0\n")
            file_handler.flush()

    file_handler.close()

if __name__ == '__main__':

    params = get_parser().parse_args()
    input_file = params.input_file
    output_file = params.output_file
    loops = params.loops
    base = params.base  # zero for raw export, else 1000
    case = params.case
    prime_encoding = params.prime_encoding
    sign_opt = params.sign_opt

    binary = (case == "binary")
    quad = (case == "quad" or case == "quad_zero")
    octuples = (case == "octuples")
    zeroes = (case == "zero" or case == "quad_zero")

    ae = (params.is_aef == 'ae')
    aef = (params.is_aef == 'aef')

    orbit = params.orbit
    sign_opt = params.sign_opt
    sign_only = (sign_opt == "sign_only")
    sign_last = (sign_opt == "sign_last")
    mag_only = (sign_opt == "mag_only")

    print(f"Reading data from {input_file} ...")
    data = convert(loops, input_file, quad, ae, aef, octuples, orbit)


    print(f"{len(data)} factors found...")
    print(f"Exporting to {output_file}.data ...")
    export_hash(data, output_file + ".data", binary, sign_only, sign_last, base, octuples, mag_only,prime_encoding)
    #generate data dict
    if params.raw_symbol_json:
        with open(f"{output_file}.data.json", "w") as outfile:
            json.dump(data, outfile)
    #
    if binary or zeroes:
        print(f"Exporting to {output_file}.zero ...")
        build_zero_file(len(data), params.trivial_frac, data, loops, output_file + '.zero', binary, sign_last, ae, aef, quad)
    print(f"Done ... ")
    if not (quad or octuples):
        print(f"cat {output_file}.data {output_file}.zero | shuf > {output_file}.prefix")
    else:
        print(f"cat {output_file}.data | shuf > {output_file}.prefix")

    print(f"To create final file ... ")
    if params.generate_relations:
        if quad:
            format = "quad"
            to_gen = rels_to_generate_compact_default
        elif case == "octuples": format = "oct"
        else:
            format = "full"
            to_gen = rels_to_generate

        print(f"generating relations ...")
        #for now, rels_to_generate is hardcoded, need to see how it varies with loop order
        output_rel_instances_jsons(loops, data, params.rels_output_path+"/", rels_to_generate=to_gen, format=format, seed=0)
        json_files = os.listdir(params.rels_output_path)
        try:
            json_files.remove("rel_instances_all_relations.json")
            print("removing old combined file")
        except:
            print("No existing combined file, creating")
        # Create an empty list to store the Python objects.
        python_objects = []
        # Load each JSON file into a Python object.
        for json_file in json_files:
            with open(params.rels_output_path+"/"+ json_file, "r") as f:
                python_objects += json.load(f)

        # Dump all the Python objects into a single JSON file.
        with open(f"{params.rels_output_path}/rel_instances_all_relations.json", "w") as f:
            json.dump(python_objects, f)