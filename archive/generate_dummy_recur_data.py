import io
import os
import sys
import math
import re
import numpy as np
from generate_data import encode, export_pairs
import random
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Language transfer")
    parser.add_argument("--recurrence_name", type=str, default="flat_evens")
    parser.add_argument("--loop_number", type=int, default=6)
    parser.add_argument("--max_skip_dist", type=int, default=2)
    parser.add_argument("--num_to_generate", type=int, default=4616466)
    parser.add_argument("--base", type=int, default=1000)

    return parser

def generate_dummy_recur_data(args):
    inputs = []
    outputs = []
    output_file = f"loop{args.loop_number}_dummydata_strike{args.max_skip_dist}_{args.recurrence_name}.data"
    num_terms = int(2*args.loop_number*args.max_skip_dist - args.max_skip_dist*(args.max_skip_dist+1)/2)
    for gen in range(args.num_to_generate):
        terms = []
        terms_encoded = []
        for my_term in range(num_terms):
            term = get_random_sign()*random.randint(0, 125)
            terms.append(term)
            terms_encoded.extend(encode(16*term,args.base))
        result = encode(get_result(terms, args.recurrence_name), args.base)
        inputs.append(terms_encoded)
        outputs.append(result)
    export_pairs(inputs, outputs, output_file)
    return

def get_random_sign():
    return 1 if random.random() < 0.5 else -1

def get_result(terms, recur_name):
    result = 0
    for index, term in enumerate(terms):
        if index % 2 == 0:
            if recur_name == 'flat_evens':
                result += term * 16
            if recur_name == 'sloped_evens':
                result += (index / 2) * term * 16
            if recur_name == "alternating":
                result += term * 16
        else:
            if recur_name == 'flat_odds':
                result += term * 16
            if recur_name == 'sloped_odds':
                result += ((index + 1) / 2) * term * 16
            if recur_name == "alternating":
                result -= term * 16
    return result

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    generate_dummy_recur_data(args)

