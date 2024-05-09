# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import math
import numpy as np
import src.envs.encoders as encoders
import src.envs.generators as generators
import json

from torch.utils.data import DataLoader
from src.dataset import EnvDataset

from ..utils import bool_flag, padic_order_equals, biggest_power_of_p

SPECIAL_WORDS = ["<eos>", "<pad>", "<sep>", "(", ")", "<MASK>"]
SPECIAL_WORDS = SPECIAL_WORDS + [f"<SPECIAL_{i}>" for i in range(10)]

logger = getLogger()

class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)

class BootsEnvironment(object):

    TRAINING_TASKS = {"boots"}
    def __init__(self, params):
        params=self.bonus_args(params)
        self.max_len = params.max_len
        self.operation = params.operation
        if self.operation != "mask":
            self.hyp_eval_metrics =["mag","sign","2adic"]
        else:
            self.hyp_eval_metrics =["matched","good","nphrases"]

        assert params.reload_data != ""
        self.eval_lookup_dict=params.eval_lookup_dict
        base = params.base
        self.modulus = params.modulus
        self.registers = params.registers
        self.append_registers = params.append_registers

        if params.prime_encoding:
            self.output_encoder = encoders.IntegerPrimeEnc(params,5,100, params.base)
            if params.numeric_import_input:
                self.input_encoder = encoders.IntegerVectorPrimeEnc(params,5,100,params.base)
        else:
            self.output_encoder = encoders.Integer(params,base)
            if params.numeric_import_input:
                self.input_encoder = encoders.IntegerVector(params, base)

        if params.word_runs:
            self.word_encoder = encoders.WordRun(params)
        else:
            self.word_encoder = encoders.WordBase(params)

        self.float_tolerance = 0.1 # params.float_tolerance
        self.additional_tolerance = []
        #    [float(x) for x in params.more_tolerance.split(",") if len(x) > 0]

        # vocabulary
        self.signs = ['+', '-']
        self.letters = ['a', 'b', 'c', 'd', 'e', 'f','g','h'] + [f"v{i:02d}" for i in range(93)]
        if params.word_runs:
            self.run_counters = [f"r{i}" for i in range(30)]
            self.letters += self.run_counters

        self.common_symbols = self.signs + self.letters
        if self.registers > 0:
            self.regwords = [f"R{i}" for i in range(params.registers)]
        else:
            self.regwords = []
        if params.numeric_import_input:
            self.words = SPECIAL_WORDS + self.common_symbols + sorted(list(
                set(self.regwords + self.output_encoder.symbols + self.input_encoder.symbols)
            ))
        else:
            self.words = SPECIAL_WORDS + self.common_symbols + sorted(list(
                set(self.regwords + self.output_encoder.symbols)
            ))
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        self.sep_index = params.sep_index = 2
        self.mask_index = params.mask_index = 5

        self.signs_mask_prob_train = params.signs_mask_prob_train
        self.letter_mask_prob_train = params.letter_mask_prob_train
        self.num_mask_prob_train = params.num_mask_prob_train
        self.signs_mask_prob_eval = params.signs_mask_prob_eval
        self.letter_mask_prob_eval = params.letter_mask_prob_eval
        self.num_mask_prob_eval = params.num_mask_prob_eval

        logger.info(f"words: {self.word2id}")

        try:
            with open(params.eval_lookup_dict, "r") as jsonFile:
                self.eval_lookup_dict = json.load(jsonFile)
        except FileNotFoundError:
            self.eval_lookup_dict = {}
            pass

    def input_to_infix(self, lst):
        return ''.join(lst)

    def decode_coef_base_1000(self,lst):
        start_idx = lst.index('<s>') + 1
        # find closing token
        end_idx = lst.index('<s>', start_idx)
        # output sublist
        strippedlist= lst[start_idx:end_idx]
        if len(strippedlist) < 3: return ''.join(strippedlist)
        elif len(strippedlist) >= 3:
            newlist = []
            for count, token in enumerate(strippedlist):
                if count >= 2:
                    if len(token) == 3:
                        token = "0" + token
                    elif len(token) == 2:
                        token = "0" + token
                    elif len(token) == 1:
                        token = "00" + token
                    else: raise ValueError
                newlist.append(token)
            return ''.join(newlist)

    def output_to_infix(self, lst):
        if self.operation == "coeffs": 
            m = self.output_encoder.decode(lst)
        elif self.operation == "mask":
            m = self.input_to_infix(lst)
        else:
            m = int(lst[0])
        if m is None:
            return lst
        return str(m)

    def idx_to_infix(self, idx, input=True):
        """
        Convert an indexed prefix expression to SymPy.
        """
        prefix = [self.id2word[wid] for wid in idx]
        infix = self.input_to_infix(prefix) if input else self.output_to_infix(prefix)
        return infix

    def gen_expr(self, data_type=None, task=None):
        """
        Generate pairs of problems and solutions.
        Encode this as a prefix sentence
        """
        # empty for now
        return None
        # gen = self.generator.generate(self.rng)
        # if gen is None:
        #     return None
        # x_data, y_data = gen
        # # encode input
        # x = self.input_encoder.encode(x_data, False)
        # # encode output
        # y = self.output_encoder.encode(y_data, True)
        # if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
        #     return None
        # return x, y

    def decode_class(self, i, do_2adic=False):
        if self.operation == "coeffs":
            if do_2adic:
                if i == 0: return str(0)
                else:
                    val = (i - 500)
                    sign = np.sign(val)
                    mag = pow(2, abs(val))
                    return str(sign*mag)
            else:
                if i == 1000: return "others"
                if i == 0: return str(0)
                else:
                    val = (i - 500)
                    sign = np.sign(val)
                    mag = pow(2, abs(val))
                    return str(sign*mag)
        #if self.operation =="mask":
        return str(i)

    def code_class(self, xi, yi, do_2adic=False):
        #index cannot be negative, so shift up
        if self.operation =="coeffs":
            nre = self.output_encoder.decode(yi)
            if do_2adic:
                #track 2adic accuracy
                if nre == 0: return 0
                else:
                    return (np.sign(nre) * math.log(biggest_power_of_p(nre,2),2) + 500)
            else:
                #track acc on all pure powers of 2
                if nre == 0: return 0
                else:
                    if math.log(np.abs(nre),2).is_integer():
                        return (np.sign(nre) * math.log(biggest_power_of_p(nre,2),2) + 500)
                    else: return 1000
        #elif self.operation =="mask":
        return int(yi[0])

    def check_prediction_lookup(self, tgt, hyp):
        # to return: fullmatch, num_match, ok, total

        if len(hyp.keys()) == 0 or len(tgt.keys()) == 0:
            return -1.0, -1.0, -1.0, -1.0
        if hyp == tgt:
            return 0.0, len(hyp.keys()), len(hyp.keys()), len(hyp.keys())

        goodcounter = 0
        matchcounter = 0
        if self.eval_lookup_dict:
            for key,val in hyp.items():
                if (key in self.eval_lookup_dict.keys()):
                    lookup = self.eval_lookup_dict[key]
                else: lookup = 0

                if val == lookup:
                    goodcounter += 1
                if (key in tgt.keys()):
                    if tgt[key] == val:
                        matchcounter += 1

        #print(hyp, tgt)
        #print(-1.0, matchcounter, goodcounter, len(hyp.keys()))
        return -1.0, matchcounter, goodcounter, len(hyp.keys())

    def check_prediction(self, src, tgt, hyp):
        if len(hyp) == 0 or len(tgt) == 0:
            return -1.0, -1.0, -1.0, -1.0
        if hyp == tgt:
            return 0.0, 0.0, 0.0, 0.0
        v = self.output_encoder.decode(tgt)
        w = self.output_encoder.decode(hyp)
        if w is None or v is None:
            return -1.0, -1.0, -1.0, -1.0
        if self.modulus > 1:
            if v < 0 or w < 0 or v >= self.modulus or w >= self.modulus:
                return -1.0, -1.0, -1.0, -1.0
        a = 0 if abs(v) == abs(w) else -1.0
        b = 0 if np.sign(v) == np.sign(w) else -1.0
        c = 0 if v == w else -1.0
        p = 0 if (padic_order_equals(v,w,2)) else -1.0
        return c, a, b, p

    def check_hypothesis(self,eq):
        """
        Check a hypothesis for a given equation and its solution.
        """
        src = [self.id2word[wid] for wid in eq["src"]]
        tgt = [self.id2word[wid] for wid in eq["tgt"]]
        hyp = [self.id2word[wid] for wid in eq["hyp"]]

        # update hypothesis
        eq["src"] = self.input_to_infix(src)
        eq["tgt"] = self.output_to_infix(tgt)
        eq["hyp"] = self.output_to_infix(hyp)
        eq["hyp_evals"] = {}

        try:
            m, s1, s2, p = self.check_prediction(src, tgt, hyp)
        except ValueError:
            m = -1.0
            s1 = -1.0
            s2 = -1.0
            p = -1.0
        #valid hyp = all correct. Universal metric
        eq["is_valid"] = m
        #task specific hyp metrics
        eq["hyp_evals"]["mag"] = s1
        eq["hyp_evals"]["sign"] = s2
        eq["hyp_evals"]["2adic"] = p
        return eq
    def list_to_keyvals(self,mylist):
        numlst = []
        charlst = []
        this_dict = {}
        # get the number. When done, save it
        # get the word part. When done, write it as a key and set
        for entry in mylist:
            if entry.isdigit() or entry == '+' or entry == '-' or ('E' in entry):
                # if we've finished a word, add it to the dict before starting this coef
                if charlst != []:
                    key = self.word_encoder.decode(charlst)
                    val = self.output_encoder.decode(numlst)
                    this_dict[''.join(key)] = val
                    charlst = []
                    numlst = []
                numlst.append(entry)
            else:
                charlst.append(entry)

        #DEBUG THIS
        key = ''.join(self.word_encoder.decode(charlst))
        val = self.output_encoder.decode(numlst)
        this_dict[''.join(key)] = val
        return this_dict
    def check_masked_hypothesis(self,eq):
        """
        Check a hypothesis for a given equation and its solution.
        """
        src = [self.id2word[wid] for wid in eq["src"]]
        tgt = [self.id2word[wid] for wid in eq["tgt"]]
        hyp = [self.id2word[wid] for wid in eq["hyp"]]

        tgt_dict = self.list_to_keyvals(tgt)
        hyp_dict = self.list_to_keyvals(hyp)

        # update hypothesis
        eq["src"] = self.input_to_infix(src)
        eq["tgt"] = self.output_to_infix(tgt)
        eq["hyp"] = self.output_to_infix(hyp)
        eq["hyp_evals"] = {}

        #do ALL key/val pairs match the tgt completely?
        #~how many~ key/val pairs match the tgt completely?
        #~how many~ key/val pairs are valid?

        try:
            is_fullmatch, num_match, num_ok, num_phrases = self.check_prediction_lookup(tgt_dict, hyp_dict)
        except ValueError:
            is_fullmatch = -1.0
            num_match = -1.0
            num_ok = -1.0
            num_phrases = -1.0

        #valid hyp = all correct. Universal metric
        eq["is_valid"] = is_fullmatch
        #task specific hyp metrics
        eq["hyp_evals"]["matched"] = num_match
        eq["hyp_evals"]["good"] = num_ok
        eq["hyp_evals"]["nphrases"] = num_phrases
        return eq


    def create_train_iterator(self, task, data_path, params):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size
    ):
        """
        Create a dataset for this environment.
        """
        assert data_type in ["valid", "test", "relations"]
        logger.info(f"Creating {data_type} iterator for {task} ...")
        if (data_type == "relations") or (data_path is None):
            mypath = data_path
        else:
            mypath = data_path[task][1 if data_type == "valid" else 2]

        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=mypath,
            size=size,
            type=data_type,
        )

        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def make_masks(self,x, train, mask_index):
        """
        x is a list of input tokens. Apply a random mask
        """
        x_new = x.copy()
        if train:
            flag=False
            for i, sample in enumerate(x_new):
                if sample in self.signs:
                    mask_prob = self.signs_mask_prob_train
                    flag = False
                elif sample in self.letters:
                    mask_prob = self.letter_mask_prob_train
                    flag = False
                elif sample in self.output_encoder.symbols:
                    mask_prob = self.num_mask_prob_train
                    if (i > 0 and x_new[i-1] in self.output_encoder.symbols) and flag == True:
                        mask_prob = 1
                else:
                    mask_prob = 0
                if np.random.random() < mask_prob:
                    # some percent of the time, replace with [MASK] token
                    x_new[i] = mask_index
                    flag = True
        else:
            flag=False
            for i, sample in enumerate(x_new):
                if sample in self.signs:
                    mask_prob = self.signs_mask_prob_eval
                    flag = False
                elif sample in self.letters:
                    mask_prob = self.letter_mask_prob_eval
                    flag = False
                elif sample in self.output_encoder.symbols:
                    mask_prob = self.num_mask_prob_eval
                    if (i > 0 and x_new[i-1] in self.output_encoder.symbols) and flag == True:
                        #if it's a number and we masked out the previous number token, mask it too
                        mask_prob = 1
                else:
                    mask_prob = 0
                if np.random.random() < mask_prob:
                    # some percent of the time, replace with [MASK] token
                    x_new[i] = mask_index
                    flag = True
        return x_new

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--operation", type=str, default="coeffs", choices=['coeffs', 'mask'],
            help="Operations to be performed: predict coeffs"
        )
        parser.add_argument("--sign_last", type=bool_flag, default=False, help="Sign as last token.")

        parser.add_argument("--mask_prob", type=float, default=0.15,
                            help="Default probability a given sign, letter, or number token will be masked, if doing masked training")

        parser.add_argument("--signs_mask_prob_train", type=float, default=None, help="Probability a given sign token will be masked, if doing masked training")
        parser.add_argument("--num_mask_prob_train", type=float, default=None, help="Probability a given num token will be masked, if doing masked training")
        parser.add_argument("--letter_mask_prob_train", type=float, default=None, help="Probability a given letter token will be masked, if doing masked training")
        parser.add_argument("--signs_mask_prob_eval", type=float, default=None, help="Probability a given sign token will be masked for the eval task, if doing masked training")
        parser.add_argument("--num_mask_prob_eval", type=float, default=None, help="Probability a given num token will be masked for the eval task, if doing masked training")
        parser.add_argument("--letter_mask_prob_eval", type=float, default=None, help="Probability a given letter token will be masked for the eval task, if doing masked training")

        parser.add_argument("--numeric_import_input", type=bool_flag, default=False, help="Input as raw ints.")
        parser.add_argument("--numeric_import_output", type=bool_flag, default=False, help="Output as raw int(s).")

        parser.add_argument("--prime_encoding", type=bool_flag, default=False,
                            help="Do prime encoding")
        parser.add_argument(
            "--biggest_prime_index", type=int, default=1, help="If prime enc, only lead with the ith prime- i.e. if "
                                                               "this is set to 1, encode as 2^N*(remainder)"
        )
        parser.add_argument(
            "--biggest_prime_power", type=int, default=1000, help="If prime enc, cap largest power at some value and put"
                                                                  "the rest in remainder"
        )
        parser.add_argument(
            "--base", type=int, default=1000, help="encoding base"
        )

        parser.add_argument(
            "--float_tolerance",
            type=float,
            default=0.1,
            help="error tolerance for float results",
        )
        parser.add_argument(
            "--more_tolerance", type=str, default="", help="additional tolerance limits"
        )

        parser.add_argument(
            "--modulus", type=int, default=0, help="modulus, 0: no modulus"
        )

        parser.add_argument(
            "--word_runs", type=bool_flag, default=False, help="encode words using 'run' notation"
        )

        parser.add_argument("--eval_lookup_dict", type=str, default="",
                            help="For objectives like masking, there may be multiple correct options.Use this to do a lookup when running check_hyp to see if ANY valid solution is returned")

    def bonus_args(self,params):
        if not params.signs_mask_prob_train:
            params.signs_mask_prob_train = params.mask_prob
        if not params.letter_mask_prob_train:
            params.letter_mask_prob_train = params.mask_prob
        if not params.num_mask_prob_train:
            params.num_mask_prob_train = params.mask_prob

        if not params.signs_mask_prob_eval:
            params.signs_mask_prob_eval = params.mask_prob
        if not params.letter_mask_prob_eval:
            params.letter_mask_prob_eval = params.mask_prob
        if not params.num_mask_prob_eval:
            params.num_mask_prob_eval = params.mask_prob

        return params