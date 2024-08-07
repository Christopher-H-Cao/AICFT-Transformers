import io
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
<<<<<<< HEAD
from rels_utils import get_dihedral_images, get_rel_instances_in_symb, replace_trivial0_terms
=======
#from rels_utils import get_dihedral_images, get_rel_instances_in_symb, replace_trivial0_terms
>>>>>>> ad6217c (clean up)
from src.utils import bool_flag, initialize_exp
from src.envs.encoders import Rational
from src.envs.cfts import CFTEnvironment
#from train import get_parser
import json
import numpy as np
from collections import Counter, OrderedDict
import itertools
from itertools import islice
import argparse

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="data generator")

    # main parameters
    parser.add_argument("--path_to_csv", type=str, default=None,
                        help="full path to input csv")
    parser.add_argument("--path_to_outfile", type=str, default = "./su2_cfts.data", help = "where to write the data?")
    parser.add_argument("--num_rows", type=int, default=-1, help="how many rows to take from the csv?")
    parser.add_argument("--num_cols", type=int, default=-1, help="how many terms from each row?")
    parser.add_argument("--target_variable", type=str, default='cc', help="target variable?")
    parser.add_argument("--target_first", type=bool_flag, default=False, help="is the target variable given first or last?")
    
    return parser

def get_ade(k):
    #This is incorrect, need input from Chris
    ade=[]
    if k >= 0: ade += 'a'
    if k >= 4 and k % 4 == 0: ade += 'd'
    if k >= 6 and (k+2) % 4 == 0: ade += 'd'
    if (k == 10 or k==16 or k==28): ade += 'e'
    return ade

def export_pairs(idata, odata, outfile):
    file_handler = io.open(outfile, mode="wt", encoding="utf-8")
    for i, (k, v) in enumerate(zip(idata, odata)):
        k=[str(num) for num in k]
        v=[str(num) for num in v]
        prefix1_str = " ".join(k)
        prefix2_str = " ".join(v)
        file_handler.write(f"{i + 1}|{prefix1_str}\t{prefix2_str}\n")
        file_handler.flush()
    file_handler.close()

if __name__ == '__main__':
    params=get_parser().parse_args()
    df = pd.read_csv(params.path_to_csv)
    if params.target_first:
        target = df.iloc[:,0]
        data = df.iloc[:,1:]
    else:
        target = df.iloc[:,-1:]
        data = df.iloc[:,:-1]

    if params.num_rows > 0:
        df=df.head(params.num_rows)
    if params.num_cols > 0:
        df=df.iloc[:, : num_cols]

    #if params.target_variable == 'ade':
    #    df['ade']=target
    if params.target_variable == 'cc':
        df['cc'] = target
    else:
        print('Error! Unknown target variable!')
        raise ValueError

    idata = data.values.tolist()
    odata = target.values.tolist()
    export_pairs(idata,odata, params.path_to_outfile)
<<<<<<< HEAD
    print(f"Please shuffle the data with \"shuf {params.path_to_outfile} > {params.path_to_outfile}\"")
=======

    print(f"Please shuffle the data with shuf {params.path_to_outfile} --output {params.path_to_outfile}")
>>>>>>> ad6217c (clean up)
