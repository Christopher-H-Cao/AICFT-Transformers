# script to generate json files of a user-defined set of linear relations matched with a given symbol
# script to evaluate various linear relations satisfied by the symbols of the 3-point form factor of phi2

import io
import os
import sys
import math
import re
import numpy as np
import time
import datetime
import itertools
from itertools import permutations
import random
import json
##############################################################################################
# AUXILIARY FUNCTIONS #
##############################################################################################


# fixed alphabet
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
quad_prefix = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

rels_to_generate= {"first": [[500]*3,[0]*3],
                            "double": [[500]*3,[0]*3],
                            "triple": [[500],[1]],
                            "dihedral": [[500],[1]],
                            "final": [[500]*19+[500]*10,[0]*19+[1]*10],
                            "integral": [[500]*3,[1]*3]
                              }

rels_to_generate_compact_default = {'first': [[500]*3, [0]*3],
                                    'double': [[500]*3, [0]*3],
                                    'triple': [[500], [1]],
                                    'integral': [[500]*3, [1]*3]}

def output_rel_instances_jsons(loop, symb, outpath, rels_to_generate, format={'full', 'quad', 'oct'}, seed=0):
    '''
    Output json files of generated relation instances.
    ---------
    INPUTS:
    loop: int; loop order.
    symb: dict; usually truth symbol, so that this function is independent of any model.
    outpath: str; folder path for output json files
    rels_to_generate: dict; format as the default `rels_to_generate_default`.   
    seed: int; random number generating seed; default 0.

    OUTPUTS:
    json files in outpath folder, one for a specific relation named in the format `rel_instances_first0.json`.
    '''
    print('Relation Generation starts at: ', datetime.datetime.now())
    
    for rel_key, rel_info in rels_to_generate.items():
        
        if rel_key == 'first':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(first_entry_rel_table[i], rel_slot=0, loop=loop, ninstance=rel_info[0][i], symb=symb, min_overlap=rel_info[1][i], format=format, seed= seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath+'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('First entry relations done at: ', datetime.datetime.now()) 

        if rel_key == 'initial':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(initial_entries_rel_table[i], rel_slot=0, loop=loop, ninstance=rel_info[0][i], symb=symb, min_overlap=rel_info[1][i], format=format, seed= seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath+'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('Multiple-initial entry relations done at: ', datetime.datetime.now())

        
        if rel_key == 'double':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(double_adjacency_rel_table[i], rel_slot=None, loop=loop, ninstance=rel_info[0][i], symb=symb, min_overlap=rel_info[1][i], format=format, seed= seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath+'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('Double adjacency relations done at: ', datetime.datetime.now()) 
    
        
        if rel_key == 'triple':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(triple_adjacency_rel_table[i], rel_slot=None, loop=loop, ninstance=rel_info[0][i], symb=symb, min_overlap=rel_info[1][i], format=format, seed= seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath+'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('Triple adjacency relations done at: ', datetime.datetime.now()) 
    
        
        if rel_key == 'integral':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(integral_rel_table[i], rel_slot=None, loop=loop, ninstance=rel_info[0][i], symb=symb, min_overlap=rel_info[1][i], format=format, seed= seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath+'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('Integrability relations done at: ', datetime.datetime.now()) 
        
        
        if rel_key == 'final':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(final_entries_rel_table[i], rel_slot=-1, loop=loop, ninstance=rel_info[0][i], symb=symb, min_overlap=rel_info[1][i], format=format, seed= seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath+'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('Final entries relations done at: ', datetime.datetime.now()) 

        if rel_key == 'dihedral':
            for i in range(len(rel_info[0])):
                rel_instance_list,cycle_instance_list,flip_instance_list = generate_dihedral_rel_instances(loop=loop, ninstance=rel_info[0][i], symb=symb, min_overlap=rel_info[1][i], format=format, seed= seed)
                with open(outpath+'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_list, f)
                with open(outpath + 'rel_instances_{}{}.json'.format("cycle", i), 'w') as f:
                    json.dump(cycle_instance_list, f)
                with open(outpath + 'rel_instances_{}{}.json'.format("flip", i), 'w') as f:
                    json.dump(flip_instance_list, f)
            print('Dihedral relations done at: ', datetime.datetime.now())

    return None

def generate_random_word(word_length, format={'full', 'quad', 'oct'}, seed=0):
    '''
    Generate a random word with a specific length.
    ---------
    INPUTS:
    word_length: int; number of letters in the generated word.
    seed: int; random number generating seed; default 0.

    OUTPUTS:
    word: str; a word with letters in the alphabet and the specific length.
    '''
    word = ''
    for i in range(word_length):
        random.seed(seed + i)
        word += alphabet[random.randint(0, len(alphabet) - 1)]
    if format == 'full':
        return word
    if format == 'quad':
        print(word,quad_prefix)
        random.seed((seed+100)*10) # must generate another random number
        prefix = quad_prefix[random.randint(0, len(quad_prefix)-1)]
        return prefix+word
    if format == 'oct': # not yet implemented
        #random.seed((seed+1000)*100) # must generate another random number
        #prefix = oct_prefix[random.randint(0, len(quad_prefix)-1)]
        #return prefix+word
        return None




def get_coeff_from_word(word, symb):
    '''
    Get the coeff of a given word in a symbol.
    ---------
    INPUTS:
    word: str; a string of letters such as 'aaae'.
    symb: dict; a dictionary with word as key, and coeff as value, e.g., {'aaae':16, 'aaaf':16}.

    OUTPUTS:
    coeff: float; if word does not exist in symb, then return 0.
    '''
    if word in symb.keys():
        return symb[word]
    return 0


def get_word_from_coeff(coeff, symb):
    '''
    Get the words corresponding to a given coeff in a symbol.
    ---------
    INPUTS:
    coeff: float; e.g., 16.
    symb: dict; a dictionary with word as key, and coeff as value, e.g., {'aaae':16, 'aaaf':16}.

    OUTPUTS:
    word: set; a set of words (strings) with the given coeff;
          if coeff does not exist in symb, then return an empty string.
    '''
    if coeff in symb.values():
        word = {i for i in symb if symb[i] == coeff}  # set
        return word
    return str()


def is_word(word, format={'full', 'quad', 'oct'}):
    '''
    Check if all the letters in a given word are all from the alphabet.
    ---------
    INPUTS:
    word: str.

    OUTPUTS:
    True/False: bool.
    '''
    if format == 'full':
        for letter in word:
            if letter not in alphabet:
                return False
        return True

    if format == 'quad':
        if word[0] not in quad_prefix:
            return False
        else:
            for letter in word[1:]:
                if letter not in alphabet:
                    return False
            return True

    if format == 'oct': # not yet implemented
        #if word[0] not in oct_prefix:
            #return False
        #else:
            #for letter in word[1:]:
                #if letter not in alphabet:
                    #return False
            #return True
        return None

def find_nonwords(symb, format={'full', 'quad', 'oct'}):
    '''
    Find all the nonwords in a given symbol;
    nonwords defined as words consisting of illegitimate letters.
    ---------
    INPUTS:
    symb: dict.

    OUTPUTS:
    nonwords: dict; dict for nonwords with {'word': coeff} in the given symbol.
    '''
    nonwords = dict()
    for word in symb.keys():
        if not is_word(word, format=format):
            coeff = get_coeff_from_word(word, symb)
            nonwords.update({word: coeff})
    return nonwords


def find_noncoeffs(symb):
    '''
    Find all the noncoeffs in a given symbol;
    noncoeffs defined as non integers.
    ---------
    INPUTS:
    symb: dict.

    OUTPUTS:
    noncoeffs: dict; dict for noncoeffs with {'word': coeff} in the given symbol.
    '''
    noncoeffs = dict()
    for coeff in symb.values():
        if not isinstance(coeff, int):
            for word in get_word_from_coeff(coeff, symb):
                noncoeffs.update({word: coeff})
    return noncoeffs


def find_nonterms(symb, format={'full', 'quad', 'oct'}):
    '''
    Find all the nonterms in a given symbol;
    nonterms defined as either the words are nonwords, or the coeffs are noncoeffs, or both.
    ---------
    INPUTS:
    symb: dict.

    OUTPUTS:
    nonterms: dict; dict for nonterms with {'word': coeff} in the given symbol.
    '''
    nonterms = dict()
    for word in symb.keys():
        coeff = symb[word]

        if not is_word(word, format=format):
            coeff = get_coeff_from_word(word, symb)
            nonterms.update({word: coeff})

        if not isinstance(coeff, int):
            for word in get_word_from_coeff(coeff, symb):
                nonterms.update({word: coeff})

    return nonterms


def count_nonterms(symb, format={'full', 'quad', 'oct'}):
    '''
    Count the number of nonterms in a given symbol.
    ---------
    INPUTS:
    symb: dict.

    OUTPUTS:
    percent: float; between 0 and 1 (0: all terms are valid).
    '''
    return len(find_nonterms(symb, format=format)) / len(symb)


def remove_nonterms(symb,format={'full', 'quad', 'oct'}):
    '''
    Remove nonterms from a given symbol.
    ---------
    INPUTS:
    symb: dict.

    OUTPUTS:
    symb: dict; all nonterms removed.
    '''
    nonterms = find_nonterms(symb,format=format)
    for key in nonterms.keys():
        symb.pop(key)
    return symb


##############################################################################################
# FROM NOW ON, ASSUME ALL INPUT SYMBOLS ARE VALID. #
##############################################################################################


##############################################################################################
# DIHEDRAL SYMMETRY #
##############################################################################################
# Dihedral symmetry is only meaningful to check with the full data format. If symbol words are
# represented in the compact formats of quad and oct, dihedral symmetry is already baked in,
# and cannot be checked, as a dihedral rotation may take us outside the given quad/oct symbol.
#
# E.g., 'acddc', which spelled out in full is 'cddcdddd', can become 'aeeaeeee' upon
# a dihedral rotation. But aeeaeeee is never in the given symb_quad in the first place,
# as there is no quad letter to designate the suffice 'eeee'.
#
# Therefore, in this section, we assume all words are given in the full format.

# a fixed look-up table for dihedral transformations

def get_dihedral_images(word):
    '''
    Get all the dihedral images of a given word.
    ---------
    INPUTS:
    word: str.

    OUTPUTS:
    dihedral_images: list; each item in the list is a word (str); always has six items.
    '''
    word_idx = [alphabet.index(l) for l in [*word]]
    dihedral_images = [''.join([dihedral_table[row][idx] for idx in word_idx]) for row in range(len(alphabet))]
    return dihedral_images

def get_cycle_images(word):
    '''
    Get all the 3-cycle dihedral images of a given word.
    ---------
    INPUTS:
    word: str.

    OUTPUTS:
    cycle_images: list; each item in the list is a word (str); always has six items.
    '''
    word_idx = [alphabet.index(l) for l in [*word]]
    cycle_images = [''.join([cycle_table[row][idx] for idx in word_idx]) for row in range(int(len(alphabet)/2))]
    return cycle_images

def get_dihedral_terms_in_symb(word, symb, count_coeffs=False):
    '''
    Get the {word: coeff} of all the dihedral images of a given word in a symbol;
    if word (or its images) is not in the symb, then coeff=0.
    ---------
    INPUTS:
    word: str.
    symb: dict.
    count_coeffs: bool; whether to output the number of occurances of a certain coeff among all dihedral images of a word;
                  default False.

    OUTPUTS:
    images_coeffs: dict; all terms dihedrally related to the given word in the symbol.
    unique_coeffs_counts: dict; number of occurances of each unique coeff among the six dihedral images of a word;
                          e.g., {'16' (coeff): 6 (# of occurances)}; return only if count_coeffs=True.
    '''
    images = get_dihedral_images(word)
    images_coeffs = dict()
    for image in images:
        images_coeffs.update({image: get_coeff_from_word(image, symb)})

    if not count_coeffs:
        return images_coeffs

    unique_coeffs, counts = np.unique(np.array(list(images_coeffs.values())), return_counts=True)
    return images_coeffs, dict(zip(unique_coeffs, counts))

def get_cycles_flips_terms_in_symb(word, symb, count_coeffs=False):
    '''
    Get the {word: coeff} of all the dihedral images of a given word in a symbol;
    if word (or its images) is not in the symb, then coeff=0.
    ---------
    INPUTS:
    word: str.
    symb: dict.
    count_coeffs: bool; whether to output the number of occurances of a certain coeff among all dihedral images of a word;
                  default False.

    OUTPUTS:
    images_coeffs: dict; all terms dihedrally related to the given word in the symbol.
    unique_coeffs_counts: dict; number of occurances of each unique coeff among the six dihedral images of a word;
                          e.g., {'16' (coeff): 6 (# of occurances)}; return only if count_coeffs=True.
    '''
    images = get_dihedral_images(word)
    cycles = get_cycle_images(word)

    images_coeffs = dict()
    cycles_coeffs = dict()
    flips_coeffs = dict()

    for image in images:
        images_coeffs.update({image: get_coeff_from_word(image, symb)})
        if image == word:
            cycles_coeffs.update({image: get_coeff_from_word(image, symb)})
            flips_coeffs.update({image: get_coeff_from_word(image, symb)})
        elif image in cycles:
            cycles_coeffs.update({image: get_coeff_from_word(image, symb)})
        else:
            flips_coeffs.update({image: get_coeff_from_word(image, symb)})
    if not count_coeffs:
        return images_coeffs, cycles_coeffs, flips_coeffs

    unique_coeffs, counts = np.unique(np.array(list(images_coeffs.values())), return_counts=True)
    return images_coeffs, cycles_coeffs, flips_coeffs,dict(zip(unique_coeffs, counts))


def count_wrong_dihedral(word, coeff_truth, symb, return_wrong_dihedral=False):
    '''
    Get the {word: coeff} of all the dihedral images of a given word in a symbol;
    if word (or its images) is not in the symb, then coeff=0.
    ---------
    INPUTS:
    word: str.
    coeff_truth: int; the ground truth value of the coeff for all dihedral images of the given word.
    symb: dict.
    return_wrong_dihedral: bool; whether to return the dict of dihedrally related words with wrong coeff;
                           default False.

    OUTPUTS:
    percent: float; percent of wrong coeffs; between 0 and 1.
    wrong_dihedral: dict; {word: coeff} where words are dihedrally related to the given word but with wrong coeff;
                    return only if return_wrong_dihedral=True.
    '''
    images_coeffs, coeff_counts = get_dihedral_terms_in_symb(word, symb, count_coeffs=True)
    wrong_dihedral = dict()
    for coeff in coeff_counts.keys():
        if coeff != coeff_truth:
            for word in get_word_from_coeff(coeff, images_coeffs):
                wrong_dihedral.update({word: coeff})

    if not return_wrong_dihedral:
        return len(wrong_dihedral) / len(images_coeffs)

    return len(wrong_dihedral) / len(images_coeffs), wrong_dihedral

##############################################################################################
# HOMOGENOUS LINEAR RELATIONS LOOK-UP TABLES#
##############################################################################################



dihedral_table = [list(permutations(alphabet[:3]))[i]+list(permutations(alphabet[3:]))[i] for i in range(len(alphabet))]
cycle_table=[dihedral_table[i] for i in [0,3,4]]
flip_table=[dihedral_table[i] for i in [0,1,2,5]]

# first entry condition
first_entry_rel_table = [{'d': 1}, {'e': 1}, {'f': 1}]  # Sec 3.1 (iv)

# double-adjacency condition: plus dihedral symmetry; any slot
double_adjacency_rel_table = [{'de': 1}, {'ad': 1}, {'da': 1}]  # eq 2.19, 2.20

# triple-adjacency relation: plus dihedral symmetry; any slot
triple_adjacency_rel_table = [{'aab': 1, 'abb': 1, 'acb': 1}]  # eq 2.21

# integrability relations: any slot
integral_rel_table = [{'ab': 1, 'ac': 1, 'ba': -1, 'ca': -1},  # eq 3.6
                      {'ca': 1, 'cb': 1, 'ac': -1, 'bc': -1},  # eq 3.7
                      {'db': 1, 'dc': -1, 'bd': -1, 'cd': 1, 'ec': 1, 'ea': -1, 'ce': -1,
                       'ae': 1, 'fa': 1, 'fb': -1, 'af': -1, 'bf': 1, 'cb': 2,
                       'bc': -2}]  # eq 3.8 coeff (8)! Takes longest time.

# multi-final-entries relations: plus dihedral symmetry
# new order: one-term relations, short relations (<=4 terms), long relations (>4 terms).
final_entries_rel_table = [{'a': 1}, {'b': 1}, {'c': 1},  # eq 4.6 (idx: 0-2)
                           {'ad': 1}, {'ed': 1},  # eq 4.7 (1) (idx: 3-4)
                           {'add': 1}, {'abd': 1}, {'ace': 1}, {'ebd': 1}, {'edd': 1},  # eq 4.9 (idx: 5-9)
                           {'addd': 1}, {'abbd': 1}, {'adbd': 1}, {'cbbd': 1},  # eq 4.10 (idx: 10-13)
                           {'ebbd': 1}, {'ebdd': 1}, {'edbd': 1}, {'eddd': 1}, {'fdbd': 1},  # eq 4.11 (idx: 14-18)

                           {'bf': 1, 'bd': -1},  # eq 4.7 (2) (idx: 19)
                           {'cdd': 1, 'cee': 1},  # eq 4.8 (1) (idx: 20)
                           {'ddbd': 1, 'dbdd': -1},  # eq 4.12 (idx: 21)
                           {'cbdd': 1, 'cdbd': -1},  # eq 4.15(1) (idx: 22)
                           {'fbd': 1, 'dbd': -1, 'bdd': 1},  # eq 4.8 (2) (idx: 23)
                           {'bddd': 1, 'faff': 1, 'dbdd': -1, 'eaff': -1, 'fbdd': 1, 'aeee': -1},  # eq 4.13 (idx: 24)
                           {'abdd': 1, 'cddd': -1 / 2, 'dcee': -1 / 2, 'aeee': 1 / 2, 'eaff': 1 / 2, 'faff': -1 / 2,
                            'ecee': 1 / 2},  # eq 4.14 (idx: 25)
                           {'cbdd': 1, 'bfff': -1 / 2, 'dcee': 1 / 2, 'ecee': -1 / 2, 'cddd': 1 / 2, 'dbdd': 1 / 2,
                            'fbdd': -1 / 2},  # eq 4.15(2) (idx: 26)
                           {'cdbd': 1, 'bfff': -1 / 2, 'dcee': 1 / 2, 'ecee': -1 / 2, 'cddd': 1 / 2, 'dbdd': 1 / 2,
                            'fbdd': -1 / 2},  # eq 4.15(3) (idx: 27)
                           {'fbbd': 1, 'dbbd': -1, 'bbdd': 1, 'faff': -1 / 2, 'dbdd': 1 / 2, 'fbdd': -1 / 2,
                            'eaff': 1 / 2, 'aeee': 1 / 2, 'bfff': -1 / 2}]  # eq 4.16 (idx: 28)

# multi-initial-entries relations: plus dihedral symmetry
# new order: one-term relations, short relations (<=4 terms), long relations (>4 terms).
initial_entries_rel_table = [{'ad': 1},
                           {'aad': 1},{'bcf': 1},{'bde': 1},{'bdf': 1},{'bda': 1},{'abd': 1},
                           {'cb': 1, 'bc': -1},
                           {'cd': 1, 'bd': -1},
                           {'aaf': 1, 'bbf': 1, 'abf': -1},
                           {'aab': 1,'aac': 1,'cca': 1,'bba': -1,'aba': -1},
                           {'bba': 1,'bbc': 1,'ccb': 1,'aab': -1,'abb': -1},
                           {'abc': 1,'aac': 1,'bbc': 1,'cca': 1,'ccb': 1},
                           {'aac': 1,'cca': 1,'bbc': -1,'ccb': -1,'afa': 1 / 2,'aaf': -1 / 2,'bbf': 1 / 2,'afb': -1 / 2}]

##############################################################################################
# GET DIHEDRAL IMAGES OF RELATIONS #
##############################################################################################


def get_rel_dihedral(rel):
    '''
    Given a relation, output all its dihedral images, where duplicated relations are removed.
    ---------
    INPUTS:
    rel: dict; one input relation; e.g. {'aab':1, 'abb':1, 'acb':1}.

    OUTPUTS:
    unique_rel_dihedral: list; each item in the list is a dict corresponding to the dihedral images of the given relation;
                  always has six items and in the fixed order of the dihedral look-up table.
    '''
    nterm = len(rel)
    term_list = list(rel.keys())
    rel_dihedral = []

    term_list_dihedral = [get_dihedral_images(term) for term in term_list]
    for i in range(len(term_list_dihedral[0])):
        rel_dihedral_term = dict()

        for iterm in range(nterm):
            rel_dihedral_term.update({term_list_dihedral[iterm][i]: rel[term_list[iterm]]})

        rel_dihedral.append(rel_dihedral_term)

    unique_rel_dihedral = []
    seen_rel_dihedral = set()

    for d in rel_dihedral:
        dict_tuple = tuple(sorted(d.items()))
        if dict_tuple not in seen_rel_dihedral:
            seen_rel_dihedral.add(dict_tuple)
            unique_rel_dihedral.append(d)

    return unique_rel_dihedral


def get_rel_table_dihedral(rel_table):
    '''
    Given a relation table, output all the dihedral images of each relation,
    where duplicated relations are removed.
    ---------
    INPUTS:
    rel_table: list of dicts; e.g., one specific linear relations look-up table.

    OUTPUTS:
    unique_rel_table_dihedral: list of dicts; each item in the list is a dict corresponding to the dihedral images of one relation in the table;
                        total length of the list: nterms of the original table * 6 - duplicated terms.
    '''
    rel_table_dihedral = []

    for rel in rel_table:
        nterm = len(rel)  # number of terms in the relation
        term_list = list(rel.keys())

        term_list_dihedral = [get_dihedral_images(term) for term in term_list]
        for i in range(len(term_list_dihedral[0])):
            rel_dihedral = dict()
            for iterm in range(nterm):
                rel_dihedral.update({term_list_dihedral[iterm][i]: rel[term_list[iterm]]})

            rel_table_dihedral.append(rel_dihedral)

        unique_rel_table_dihedral = []
        seen_rel_table_dihedral = set()

        for d in rel_table_dihedral:
            dict_tuple = tuple(sorted(d.items()))
            if dict_tuple not in seen_rel_table_dihedral:
                seen_rel_table_dihedral.add(dict_tuple)
                unique_rel_table_dihedral.append(d)

    return unique_rel_table_dihedral


##############################################################################################
# GENERATE RELATION INSTANCES #
##############################################################################################


def generate_rel_instances(rel, rel_slot, loop, ninstance, symb=None, min_overlap=0, format={'full', 'quad', 'oct'}, seed=0):
    '''
    Given a number of random relation instances for a given relation.
    If using default, then the generated relation instances are completely independent of any symb.
    If there is input symb and min_overlap >0, then we need at least min_overlap number of words
    in the relation instances to also occur in the given symb; otherwise, reject the generated instances.
    ---------
    INPUTS:
    rel: dict; one specific linear relation in the relation look-up tables.
    rel_slot: int and None; slot in which the relation keywords are to be inserted; between 0 and inf;
              -1 for placing relation keywords at the end, i.e., for final entries relations;
              and None for any slot.
    loop: int; loop order of the generated words.
    ninstance: int; number of relation instances to be generated.
    symb: dict; input symbol to boost efficiency, shoud be truth symbol at a given loop; default None.
    min_overlap: int; min number of overlapping words occuring both in the relation relation and in
                 the input symbol in order to accept the generated instance; default 0; only meaningful
                 if symb is not None.
    seed: int; random number generating seed; default 0.

    OUTPUTS:
    rel_instance_list: list of dicts; each item in the list is a dict corresponding to one relation instance,
                       where the key is a randomly generated word realizing the specific relation instance,
                       the value is the corresponding rel coeff, and the relation keywords are all placed at
                       the pre-specified rel_slot.
    '''

    nterm_rel = len(rel)  # number of terms in the rel
    nletter_rel = len(list(rel.keys())[0])  # number of letters in each term in the rel
    if format == 'full':
        nletter_subword = 2*loop-nletter_rel
    if format == 'quad':
        nletter_subword = 2*loop-4-nletter_rel
    if format == 'oct':
        nletter_subword = 2*loop-8-nletter_rel
    rel_instance_list = []
    max_trial = 1000000
    n_trial = 0
    while len(rel_instance_list) < ninstance:
        n_trial += 1
        if n_trial > max_trial:
            break
        i = n_trial
        subword = generate_random_word(nletter_subword, format='full', seed=seed+nletter_subword+i+1)
        # generate random prefix for compact formats
        if format == 'quad':
            random.seed((seed+100*i)*10)
            prefix = quad_prefix[random.randint(0, len(quad_prefix)-1)]
        if format == 'oct': # not yet implemented
            #random.seed((seed+1000*i)*100)
            #prefix = oct_prefix[random.randint(0, len(oct_prefix)-1)]
            prefix = '1'
        rel_instance = dict()

        for key, value in rel.items():
            if rel_slot == -1: # final entries relations
                if format == 'full':
                    word = subword + key
                else: # no final entries relations for compact formats
                    #word = None
                    return None

            elif rel_slot == None:
                random.seed((seed + nletter_subword + i + 1) * 100)
                rel_slot_any = random.randint(0, 2 * loop)
                if format == 'full':
                    word = subword[:rel_slot_any] + key + subword[rel_slot_any:]
                else:  # compact formats
                    word = prefix + subword[:rel_slot_any] + key + subword[rel_slot_any:]

            else:
                if format == 'full':
                    word = subword[:rel_slot] + key + subword[rel_slot:]
                else: # compact formats
                    word = prefix + subword[:rel_slot] + key + subword[rel_slot:]

            rel_instance.update({word: value})

        if symb:
            count = 0
            for word in list(rel_instance.keys()):
                if word in symb:
                    count += 1
            if count >= min_overlap:
                rel_instance_list.append(rel_instance)

        else:
            rel_instance_list.append(rel_instance)

    return rel_instance_list
def generate_dihedral_rel_instances(loop, ninstance, symb=None, min_overlap=0, format={'full'}, seed=0):
    '''
    '''
    #nterm_rel = len(rel)  # number of terms in the rel
    #nletter_rel = len(list(rel.keys())[0])  # number of letters in each term in the rel
    if format == 'full':
        nletter_subword = 2*loop
    dihedral_instance_list = []
    cycle_instance_list = []
    flip_instance_list = []

    max_trial = 1000000
    n_trial = 0
    seen_words_dihedral = {}

    while (len(cycle_instance_list) < ninstance) or (len(flip_instance_list) < ninstance):
        # pull dihedral relations directly from the symb.
        #generate a random word from the symbol and get images
        key, coef = random.choice(list(symb.items()))
        n_trial += 1
        if n_trial > max_trial:
            break
        if key in seen_words_dihedral: continue
        images_coeffs = get_dihedral_terms_in_symb(key,symb)
        #add all images to the hash table so we don't generate the same rel twice
        seen_words_dihedral = seen_words_dihedral | images_coeffs
        dihedral_instance_list,cycle_instance_list,flip_instance_list = (
            get_relpairs(images_coeffs,dihedral_instance_list,cycle_instance_list,flip_instance_list))

    return dihedral_instance_list, cycle_instance_list, flip_instance_list

def get_relpairs(images_set,dihedral_instance_list=[],cycle_instance_list=[],flip_instance_list=[]):
    '''
    Given a dict of images, get all pairs and make two-term relations. Return the set in a list given as an arg.
    ---------
    INPUTS:
    images_set: dict, set of dihedral images of a key
    rel_instance_list: the list of relation instances to append the pairs to
    OUTPUTS:
    rel_instance_list: the list of relation instances to append the pairs to, with pairs appended
    '''
    cycle1_set={0,3,4}
    cycle2_set={1,2,5}
    my_cycle_list=[]
    my_flip_list=[]
    # get all possible pairs of dihedral images, merge each pair into one two-term dict
    for a_idx, a in enumerate(list(images_set.keys())):
        for my_idx, b in enumerate(list(images_set.keys())[a_idx + 1:]):
            b_idx=my_idx+a_idx+1
            pair = {a: images_set[a], b: images_set[b]}
            #update the rel coeffs
            for idx, (word, symb_coeff) in enumerate(pair.items()):
                if idx == 0:
                    rel_coeff = 1
                else:
                    rel_coeff = -1
                pair.update({word: [symb_coeff, rel_coeff]})
            #one dict with all pairs
            dihedral_instance_list.append(pair)
            #if the pair is in a cycle or a flip
            if (a_idx in cycle1_set and b_idx in cycle1_set) or (a_idx in cycle2_set and b_idx in cycle2_set):
                my_cycle_list.append(pair)
            else:
                my_flip_list.append(pair)

    # randomly pull one instance from the set. overlaps lead to noisy curves, since we're double-counting words
    mynum = random.random()
    if mynum > 0.5:
        inst = random.choice(my_cycle_list)
        cycle_instance_list.append(inst)
        dihedral_instance_list.append(inst)
    else:
        inst = random.choice(my_flip_list)
        flip_instance_list.append(inst)
        dihedral_instance_list.append(inst)

    return dihedral_instance_list, cycle_instance_list, flip_instance_list

##############################################################################################
# GET TERMS IN SYMBOL RELATED BY CERTAIN RELATIONS #
##############################################################################################


def get_rel_instances_in_symb(rel_instance_list, symb):
    '''
    Given a list of relation instances, get their term(s) in the given symbol.
    First step for relation-level relation check.
    ---------
    INPUTS:
    rel_instance_list: list of dicts; outputs of functions `generate_rel_instances`.
    symb: dict; symbol at a given loop order; can be the ground truth or the model predictions.

    OUTPUTS:
    rel_instance_in_symb_list: list of dicts; each item in the list is a dict corresponding to one relation instance
                               in the input rel_instance_list; key is word and value is [symb_coeff, rel_coeff];
                               if the word is not in symb, then its symb_coeff is automatically 0.
    '''
    rel_instance_in_symb_list = []
    if rel_instance_list is None:
        return None

    for rel_instance in rel_instance_list:
        rel_instance_in_symb = dict()
        for word, rel_coeff in rel_instance.items():
            symb_coeff = get_coeff_from_word(word, symb)
            rel_instance_in_symb.update({word: [symb_coeff, rel_coeff]})
        rel_instance_in_symb_list.append(rel_instance_in_symb)

    return rel_instance_in_symb_list


def update_rel_instances_in_symb(rel_instance_in_symb_list, symb):
    '''
    Given a list of relation instances in the format {word: [symb_coeff, rel_coeff]},
    update symb_coeff by the new input symbol.
    ---------
    INPUTS:
    rel_instance_list: list of dicts; format: {word: [symb_coeff, rel_coeff]}.
    symb: dict; symbol at a given loop order; can be the ground truth or the model predictions.

    OUTPUTS:
    rel_instance_in_symb_list: list of dicts; same as the output of function 'get_rel_instances_in_symb';
                               each item in the list is a dict corresponding to one relation instance
                               in the input rel_instance_list; key is word and value is [symb_coeff, rel_coeff].
    '''
    rel_instance_in_symb_list_updated = []
    if rel_instance_in_symb_list is None:
        return None


    for rel_instance in rel_instance_in_symb_list:
        rel_instance_in_symb = dict()
        for word, coeff_pair in rel_instance.items():
            symb_coeff = get_coeff_from_word(word, symb)
            rel_instance_in_symb.update({word: [symb_coeff, coeff_pair[1]]})
        rel_instance_in_symb_list_updated.append(rel_instance_in_symb)

    return rel_instance_in_symb_list_updated


def get_rel_terms_in_symb_per_word(word, symb, rel, rel_slot={'first', 'initial', 'final', 'any'}, format={'full', 'quad', 'oct'}):
    '''
    Given a word, get the related term(s) in the given symbol according to the specified relation.
    This serves as an intermediate step to get related term(s) for more words in the full symbol.
    ---------
    INPUTS:
    word: str.
    symb: dict.
    rel: dict.
    rel_slot: str; one of the three choices 'first', 'final', 'any';
              if 'first' or 'final', the relation can only be at the first or final few slots of a word;
              if 'any', the relation can be at any slot of a word.

    OUTPUTS:
    rel_terms_list: list of dicts; each item in the list is a dict in the format of
                    {'bbbf' (word): [16 (coeff), 1 (rel coeff)]}.
    '''
    rel_terms_list = []
    nterm = len(rel)
    nletter = len(list(rel.keys())[0])  # number of letters in each term

    # first entry relation
    if rel_slot == 'first':
        rel_terms = dict()

        if format == 'full':
            if word[:nletter] not in rel.keys():
                return rel_terms_list
            else:
                rel_terms.update({word: [get_coeff_from_word(word, symb), rel[word[:nletter]]]})
                rel_terms_list.append(rel_terms)
                return rel_terms_list
        else: # compact formats
            if word[1:nletter] not in rel.keys(): # ignore the prefix
                return rel_terms_list
            else:
                rel_terms.update({word: [get_coeff_from_word(word, symb), rel[word[1:nletter]]]})
                rel_terms_list.append(rel_terms)
                return rel_terms_list

    # first entry relation
    if rel_slot == 'initial':
        rel_terms = dict()

        if format == 'full':
            if word[:nletter] not in rel.keys():
                return rel_terms_list
            else:
                rel_terms.update({word: [get_coeff_from_word(word, symb), rel[word[:nletter]]]})
                rel_terms_list.append(rel_terms)
                return rel_terms_list
        else: # compact formats
            if word[1:nletter] not in rel.keys(): # ignore the prefix
                return rel_terms_list
            else:
                rel_terms.update({word: [get_coeff_from_word(word, symb), rel[word[1:nletter]]]})
                rel_terms_list.append(rel_terms)
                return rel_terms_list


    # final entries relation
    if rel_slot == 'final':
        rel_terms = dict()

        if format == 'full':
            if word[-nletter:] not in rel.keys():
                    return rel_terms_list

            for key in rel.keys():
                if word[-nletter:] == key:
                    pre_subword = word[:-nletter]
                    for key_rel in rel.keys():
                        word_rel = pre_subword+key_rel
                        rel_terms.update({word_rel:[get_coeff_from_word(word_rel, symb), rel[key_rel]]})
                    rel_terms_list.append(rel_terms)
                    return rel_terms_list
        else: # compact formats: no final entries relations
            return None

    # relations that can be at any position slot
    if rel_slot == 'any':

        key_rel_list = list(rel.keys())

        if format == 'full':
            word = word
        else: # compact formats
            prefix = word[0]
            word = word[1:]


        if not any(key_rel in word for key_rel in key_rel_list):
            # rel_terms = dict()
            # rel_terms_list.append(rel_terms)
            return rel_terms_list

        for key_rel in rel.keys():

            if key_rel in word:
                start_pos_list = [i.start() for i in re.finditer(key_rel, word)]
                rel_terms_pos = []

                for start_pos in start_pos_list:
                    rel_terms = dict()

                    pre_subword = word[:start_pos]
                    post_subword = word[start_pos + nletter:]

                    for key_rel in rel.keys():
                        if format == 'full':
                            word_rel = pre_subword+key_rel+post_subword
                        else: # compact formats
                            word_rel = prefix+pre_subword+key_rel+post_subword
                        rel_terms.update({word_rel: [get_coeff_from_word(word_rel, symb), rel[key_rel]]})

                    rel_terms_pos.append(rel_terms)

                rel_terms_list.append(rel_terms_pos)

        return list(itertools.chain(*rel_terms_list))


def get_rel_terms_in_symb(symb, fraction, rel, rel_slot={'first', 'final', 'initial', 'any'}, format={'full', 'quad', 'oct'},seed=0):
    '''
    Get the related term(s) in the given symbol according to the specified relation,
    for a fraction of words in the full symbol.
    ---------
    INPUTS:
    symb: dict.
    fraction: float; fraction of total words in the full symbol to be picked as samples;
              between 0 and 1, where 1 gets all words in the full symbol;
    rel: dict.
    rel_slot: str; one of the three choices 'first', 'final', 'any';
              if 'first' or 'final', the relation can only be at the first or final few slots of a word;
              if 'any', the relation can be at any slot of a word.
    seed: int; random number generating seed; default 0.

    OUTPUTS:
    rel_terms_list: list of dicts; each item in the list is a dict in the format of
                    {word: [symb_coeff, rel_coeff]}.
    '''
    rel_terms_list_symb = []

    all_words = list(symb.keys())
    num_words_to_pick = int(len(all_words) * fraction)

    if num_words_to_pick <= 0:
        return []

    random.seed(seed)
    random_words = random.sample(all_words, num_words_to_pick)

    for word in random_words:
        rel_terms_list = get_rel_terms_in_symb_per_word(word, symb, rel, rel_slot=rel_slot, format=format)
        if rel_terms_list is None:
            return None
        for rel_terms in rel_terms_list:
            if (rel_terms) and (
            not any([rel_terms == existing_rel_terms for existing_rel_terms in rel_terms_list_symb])):
                rel_terms_list_symb.append(rel_terms)

    return rel_terms_list_symb


##############################################################################################
# CHECK RELATIONS IN SYMBOL #
##############################################################################################


def check_rel(rel_terms_list, return_rel_info=False, p_norm=None):
    '''
    Check if all relations in the input list of relations are satisfied, i.e., sum up to 0.
    Valid regardless of the sampling approach, i.e., word-oriented or relation-oriented.
    ---------
    INPUTS:
    rel_terms_list: list of dicts; format {word: [symb_coeff, rel_coeff]}.
    return_rel_info: bool; whether to return the detailed info of each relation,
                     including relation sum and number of non-trivial-zero terms in each relation;
                     default False.
    p_norm: float or None; p is the average accuracy of model prediction;
            goal is to normalize the accuracy by the number of terms in a relation; default None.

    OUTPUTS:
    percent: float; percent of correct relations, i.e., those that sum up to 0; between 0 and 1 (1: all correct).
    relsum_list: list; each item in the list is a int/float for one instance,
                 where the int/float corresponds to the rel sum of that particular instance (should all be 0 if correct);
                 return only if return_info_info=True.
    relnontrivial0_list: list; each item in the list is an int for one instance,
                 where the int is the number of non-trivial-zero words in that particular instance;
                 return only if return_rel_info=True.
    '''
    relsum_list, relnontrivial0_list = [], []

    if rel_terms_list is None:
        return None
    for rel_terms in rel_terms_list:
        relsum = 0
        n_nontrivial0_term = 0
        nterm = len(rel_terms)
        for key, value in rel_terms.items():
            if value[0] == None:  # invalid symb_coeff
                relsum = -1
            else:
                try:
                    relsum += np.prod(value)
                except ArithmeticError:
                    print("overflow error! setting rel sum to -1")
                    relsum = -1

            if not is_trivial0(key):
                n_nontrivial0_term += 1

        if p_norm:
            try:
                relsum /= p_norm ** nterm
            except ArithmeticError:
                print("overflow error! setting rel sum to -1")
                relsum = -1
        relsum_list.append(relsum)
        relnontrivial0_list.append(n_nontrivial0_term)

    if not relsum_list:
        percent = None
    else:
        percent = relsum_list.count(0) / len(relsum_list)

        if p_norm:
            percent /= p_norm ** nterm

    if return_rel_info:
        return percent, relsum_list, relnontrivial0_list

    return percent



##############################################################################################
# WORD-ORIENTED APPROACH (works at 5 loops, but not scalable!) #
##############################################################################################


# The default relations to check in the format of {rel name: [frac, ..., frac]}, 
# where frac (0-1) is the fraction of the total words in a symbol to be checked  
# and the list order follows rel ID as defined in the relation tables. 
# If some rel ID is not to be checked, then set its corresponding frac to 0.
# Can be modified according to the user's needs. 

rels_to_check_default = {'first': [0.1, 0.1, 0.1], 'double': [0.1, 0.1, 0.1], 'triple': [0.1], 
                         'final': [0.1]*29, 'integral': [0.01, 0.01, 0.01]}


def assess_rels_viaword(symb, symb_truth, rels_to_check, p_norm=None, format={'full', 'quad', 'oct'}, seed=0):
    '''
    Assess the success rate of a set of user-specified relations for a given symbol (usually model predictions),
    thru rels_to_check for word-oriented approach.
    One symbol only, which means one epoch model output.
    ---------
    INPUTS:
    symb: dict; usually model predictions.
    symb_truth: dict; truth symbol.
    rels_to_check: dict; user-defined relations to check; formats same as the two defaults defined above.
    p_norm: float or None; p is the average accuracy of model prediction;
            goal is to normalize the accuracy by the number of terms in a relation; default None.
    format: string; different formats to represent the words;
            only accept three choices---'full', 'quad', 'oct';
            if format is 'full', then can use `rels_to_check_full_default`;
            if format is 'quad' or 'oct, then can use `rels_to_check_compact_default`;
            note that here relations living across the seam are not considered, i.e.,
            only relations among exposed letters are considered; the format prefix is generated randomly.
    seed: int; random number generating seed; default 0;
          in principle, can even have a seed_list for different relations (currently not implemented).

    OUTPUTS:
    rel_acc: dict; format: {rel ID: [percent, percent_allcorrect, percent_magcorrect, percent_signcorrect, num_rels]},
                    where percent is the percent of correct relations,
                    percent_allcorrect is the percent of correct relations with all its coeffs correct,
                    percent_magcorrect is the percent of correct relations with all its coeffs mag-correct,
                    percent_signcorrect is the percent of correct relations with all its coeffs sign-correct,
                    and num_rels is the number of relation instances.
    '''
    print('Assessment via word approach starts at: ', datetime.datetime.now())
    rel_acc = dict()

    for rel_key, rel_frac_list in rels_to_check.items():

        if rel_key == 'first':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], first_entry_rel_table[i],
                                                              rel_slot='first', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'first{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                      percent_signcorrect, len(rel_term_in_symb_list)]})
            print('First entry relations done at: ', datetime.datetime.now())

        if rel_key == 'double':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], double_adjacency_rel_table[i],
                                                              rel_slot='any', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'double{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                       percent_signcorrect, len(rel_term_in_symb_list)]})
            print('Double adjacency relations done at: ', datetime.datetime.now())

        if rel_key == 'triple':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], triple_adjacency_rel_table[i],
                                                              rel_slot='any', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'triple{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                       percent_signcorrect, len(rel_term_in_symb_list)]})
            print('Triple adjacency relations done at: ', datetime.datetime.now())

        if rel_key == 'final':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], final_entries_rel_table[i],
                                                              rel_slot='final', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'final{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                      percent_signcorrect, len(rel_term_in_symb_list)]})
            print('Final entries relations done at: ', datetime.datetime.now())

        if rel_key == 'integral':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], integral_rel_table[i],
                                                              rel_slot='any', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'integral{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                         percent_signcorrect, len(rel_term_in_symb_list)]})
            print('Integrability relations done at: ', datetime.datetime.now())

        if rel_key == 'initial':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], initial_entries_rel_table[i],
                                                              rel_slot='initial', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'initial{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                      percent_signcorrect, len(rel_term_in_symb_list)]})
            print('Initial entries relations done at: ', datetime.datetime.now())
    return rel_acc
    
##############################################################################################
# RELATION-ORIENTED APPROACH #
##############################################################################################


def assess_rels_viarel(symb, symb_truth, inpath, p_norm=None):
    '''
    Assess the success rate of a set of user-specified relations for a given symbol (usually model predictions),
    thru the stored relation instances for relation-oriented approach.
    One symbol only, which means one epoch model output.
    ---------
    INPUTS:
    symb: dict; usually model predictions.
    symb_truth: dict; truth symbol.
    inpath: str; folder path for rel_data.
    p_norm: float or None; p is the average accuracy of model prediction;
            goal is to normalize the accuracy by the number of terms in a relation; default None.

    OUTPUTS:
    rel_acc: dict; format: {rel ID: [percent, percent_allcorrect, percent_magcorrect, percent_signcorrect, num_rels]},
                    where percent is the percent of correct relations,
                    percent_allcorrect is the percent of correct relations with all its coeffs correct,
                    percent_magcorrect is the percent of correct relations with all its coeffs mag-correct,
                    percent_signcorrect is the percent of correct relations with all its coeffs sign-correct,
                    and num_rels is the number of relation instances.
    '''
    print('Assessment via relation approach starts at: ', datetime.datetime.now())
    rel_acc = dict()

    for filename in os.listdir(inpath):
        print('Assessing relation file {} ...'.format(filename))
        parts = filename.split("_")
        rel_key = parts[-1].split(".")[0]

        file_path = os.path.join(inpath, filename)
        if os.path.isfile(file_path) and filename.endswith('.json'):
            with open(file_path, 'r') as openfile:
                rel_instance_list = json.load(openfile)

                rel_instance_in_symb_list = update_rel_instances_in_symb(rel_instance_list, symb)
                percent = check_rel(rel_instance_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_instance_in_symb_list, symb_truth, return_counts=False)
                rel_acc.update({rel_key: [percent, percent_allcorrect, percent_magcorrect, percent_signcorrect, len(rel_instance_in_symb_list)]})

    print('Assessment via relation approach ends at: ', datetime.datetime.now())
    return rel_acc


def check_coeffs_in_rel(rel_terms_list, symb_truth_list, return_counts=False,require_satisfied=True):
    '''
    Check if all relations in the input list of relations are satisfied, i.e., sum up to 0.
    Valid regardless of the sampling approach, i.e., word-oriented or relation-oriented.
    ---------
    INPUTS:
    rel_terms_list: list of dicts; format {word: [symb_coeff, rel_coeff]}.
    symb_truth: list of dicts;format {word: [symb_coeff, rel_coeff]}. the truth symbol against which the correctness of
                symb_coeff predicted by the model for each word is checked.
    return_counts: bool; whether to return the count of correct coeffs in each rel instance;
                    default False.

    OUTPUTS:
    percent_allcorrect: float; percent of correct relations with all its word coeffs also correct.
    percent_magcorrect: float; percent of correct relations with all its word coeffs magnitude correct.
    correct_coeffs_in_rel_list: list of lists; format [[bool, int, int], [], ...], where bool suggests if the current relation
                            instance is correct, the first int indicates how many of its word coeffs are right,
                            and the second int suggests how many word coeffs are magnitude correct;
                            return only if return_counts=True.
    '''
    correct_coeffs_in_rel_list = []
    n_allcorrect_rel, n_magcorrect_rel, n_signcorrect_rel = 0, 0, 0
    n_allcorrect_norel, n_magcorrect_norel, n_signcorrect_norel = 0, 0, 0
    if rel_terms_list is None:
        return None

    for rel_terms, symb_truth in zip(rel_terms_list,symb_truth_list):
        relsum = 0
        n_allcorrect, n_magcorrect, n_signcorrect = 0, 0, 0
        nterm = len(rel_terms)

        for (key, value), (truth_key, truth_value) in zip(rel_terms.items(),symb_truth.items()):
            if value[0] == None:  # invalid symb_coeff
                relsum = -1
            else:
                try:
                    relsum += np.prod(value)
                except ArithmeticError:
                    print("overflow error! setting rel sum to -1")
                    relsum = -1

            if value[0] != None:
                if value[0] == truth_value[0]:
                    n_allcorrect += 1

                if np.abs(value[0]) == np.abs(truth_value[0]):
                    n_magcorrect += 1

                if np.sign(value[0]) == np.sign(truth_value[0]):
                    n_signcorrect += 1

        if relsum == 0:
            rel_correct = True
        else:
            rel_correct = False

        if rel_correct == True and n_allcorrect == nterm:
            n_allcorrect_rel += 1

        if rel_correct == True and n_magcorrect == nterm:
            n_magcorrect_rel += 1

        if rel_correct == True and n_signcorrect == nterm:
            n_signcorrect_rel += 1

        if n_allcorrect == nterm:
            n_allcorrect_norel += 1

        if n_magcorrect == nterm:
            n_magcorrect_norel += 1

        if n_signcorrect == nterm:
            n_signcorrect_norel += 1

        correct_coeffs_in_rel_list.append([rel_correct, n_allcorrect, n_magcorrect, n_signcorrect])
    if not correct_coeffs_in_rel_list:
        percent_allcorrect, percent_magcorrect, percent_signcorrect = None, None, None
    else:
        if require_satisfied:
            percent_allcorrect = n_allcorrect_rel / len(correct_coeffs_in_rel_list)
            percent_magcorrect = n_magcorrect_rel / len(correct_coeffs_in_rel_list)
            percent_signcorrect = n_signcorrect_rel / len(correct_coeffs_in_rel_list)
        else:
            percent_allcorrect = n_allcorrect_norel / len(correct_coeffs_in_rel_list)
            percent_magcorrect = n_magcorrect_norel / len(correct_coeffs_in_rel_list)
            percent_signcorrect = n_signcorrect_norel / len(correct_coeffs_in_rel_list)
    if return_counts:
        return percent_allcorrect, percent_magcorrect, percent_signcorrect, correct_coeffs_in_rel_list

    return percent_allcorrect, percent_magcorrect, percent_signcorrect

##############################################################################################
# ZERO SAMPLING  #
##############################################################################################

# Assume all words are in the full format (no quad, oct).

def is_trivial0(word):
    '''
    Check if a given word (assuming valid and in full format) is a trivial zero word.
    ---------
    INPUTS:
    word: str.
    OUTPUTS:
    True/False: bool.
    '''
    for rel in first_entry_rel_table: # prefix rule
        if word[0] in rel.keys():
            return True

    for rel in final_entries_rel_table[:3]: # suffix rule
        if word[-1] in rel.keys():
            return True

    for rel in get_rel_table_dihedral(double_adjacency_rel_table): # adjacency rule
        for rel_key in rel.keys():
            if rel_key in word:
                return True

    return False



def replace_trivial0_terms(symb, return_symb=False):
    '''
    Find all the trivial zero words in a given symbol (assume valid and in full format)
    and manually force their coeffs to be zero, regardless of the original coeffs in the given symbol.
    ---------
    INPUTS:
    symb: dict.
    return_symb: bool; whether to return the full given symbol
                       with its trivial zero terms updated to have coeff=0;
                       default False.
    OUTPUTS:
    if return_symb == False, then
        trivial0_terms: dict; dict for trivial zero terms with {'word': 0} in the given symbol.
    if return_symb == True, then
        symb_updated: dict; full input symbol with its trivial zero terms updated to have coeff=0.
    '''
    trivial0_terms = dict()
    symb_updated = symb.copy()
    for word in symb.keys():
        if is_trivial0(word):
            trivial0_terms.update({word: 0})
            symb_updated.update({word: 0})

    if not return_symb:
        return trivial0_terms

    if return_symb:
        return symb_updated

def generate_trivial0_symb(nterms_pertype, loop, seed=0, format="full"):
    '''
    Generate a symbol in full format consisting solely of trivial zero terms.
    ---------
    INPUTS:
    nterms_pertype: int; number of terms per type of trivial zero;
                         the total terms in the generated symbol is nterms_pertype * n_types.
    loop: int; loop order of the generated words.
    seed: int; random number generating seed; default 0.
    OUTPUTS:
    trivial0_symb: dict; the generated symbol consisting solely of trivial zero terms.
    '''
    trivial0_symb = dict()

    for rel in first_entry_rel_table: # prefix rule
        prefix_rels = generate_rel_instances(rel, 0, loop, nterms_pertype, format=format, seed=seed)
        for prefix_rel in prefix_rels:
            for word in prefix_rel.keys():
                trivial0_symb.update({word: 0})

    if format == "full":
        for rel in final_entries_rel_table[:3]: # suffix rule
            suffix_rels = generate_rel_instances(rel, -1, loop, nterms_pertype, format=format, seed=seed+1)
            for suffix_rel in suffix_rels:
                for word in suffix_rel.keys():
                    trivial0_symb.update({word: 0})

    for rel in get_rel_table_dihedral(double_adjacency_rel_table): # adjacency rule
        adjacency_rels = generate_rel_instances(rel, None, loop, nterms_pertype, format=format, seed=seed+2)
        for adjacency_rel in adjacency_rels:
            for word in adjacency_rel.keys():
                trivial0_symb.update({word: 0})

    return trivial0_symb