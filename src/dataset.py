# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import io
import sys
import json
import numpy as np

from .utils import encode_num
import torch
from torch.utils.data.dataset import Dataset

logger = getLogger()


class EnvDataset(Dataset):
    def __init__(self, env, task, train, params, path, size=None, type=None):
        super(EnvDataset).__init__()
        self.env = env
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.prime_encoding = params.prime_encoding
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        self.type = type
        self.numeric_import_input = params.numeric_import_input
        self.numeric_import_output = params.numeric_import_output
        self.decoder_only = (params.architecture == 'decoder_only')
        assert size is None or not self.train
        assert not params.batch_load or params.reload_size > 0

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size
        self.operation = params.operation
        self.full_sentence = params.full_sentence
        self.batch_load = params.batch_load
        self.reload_size = params.reload_size
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0

        # generation, or reloading from file
        if path is not None:
            if not self.full_sentence and self.type != "relations":
                assert os.path.isfile(path)
                if params.batch_load and self.train:
                    self.load_chunk()
                else:
                    logger.info(f"Loading data from {path} ...")
                    with io.open(path, mode="r", encoding="utf-8") as f:
                        # either reload the entire file, or the first N lines
                        # (for the training set)
                        if not train:
                            lines = [line.rstrip().split("|") for line in f]
                        else:
                            lines = []
                            for i, line in enumerate(f):
                                if i == params.reload_size:
                                    break
                                if i % params.n_gpu_per_node == params.local_rank:
                                    lines.append(line.rstrip().split("|"))
                    self.data = [xy.split("\t") for _, xy in lines]
                    self.data = [xy for xy in self.data if len(xy) == 2]
                    logger.info(f"Loaded {len(self.data)} equations from the disk.")
            elif not self.full_sentence and self.type == "relations":
                #Get relations from tianji's rel files
                assert os.path.isfile(path)
                logger.info(f"Loading data from {path} ...")
                self.data = []
                with open(path) as json_data:
                    d = json.load(json_data)
                    for rel_inst in d:
                        for key, val in rel_inst.items():
                            self.data.append([key,val[0]])
                logger.info(f"Evaluating {len(self.data)} words from {len(d)} relation instances.")
            else:
                #Get relations from tianji's rel files
                assert os.path.isfile(path)
                logger.info(f"Loading data from {path} ...")
                self.data = []
                with open(path) as json_data:
                    d = json.load(json_data)
                    for rel_inst in d:
                        self.data.append(rel_inst)
                logger.info(f"Looking at {len(d)} relation instances.")
            # dataset size: infinite iterator for train, finite for valid / test
            # (default of 10000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            self.size = 10000 if path is None else len(self.data)
        else:
            assert size > 0
            self.size = size

    def load_chunk(self):
        self.basepos = self.nextpos
        logger.info(
            f"Loading data from {self.path} ... seekpos {self.seekpos}, "
            f"basepos {self.basepos}"
        )
        endfile = False
        with io.open(self.path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines = []
            for i in range(self.reload_size):
                line = f.readline()
                if not line:
                    endfile = True
                    break
                if i % self.n_gpu_per_node == self.local_rank:
                    lines.append(line.rstrip().split("|"))
            self.seekpos = 0 if endfile else f.tell()

        self.data = [xy.split("\t") for _, xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]
        self.nextpos = self.basepos + len(self.data)
        logger.info(
            f"Loaded {len(self.data)} equations from the disk. seekpos {self.seekpos}, "
            f"nextpos {self.nextpos}"
        )
        if len(self.data) == 0:
            self.load_chunk()

    def batch_sequences(self, sequences, pad_index, bos_index, eos_index):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(
            pad_index
        )
        assert lengths.min().item() > 2

        sent[0] = bos_index
        for i, s in enumerate(sequences):
            sent[1 : lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = eos_index

        return sent, lengths

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        if not self.operation=='mask':
            #"old" training
            x, y = zip(*elements)
            nb_eqs = [self.env.code_class(xi, yi) for xi, yi in zip(x, y)]
            if self.env.registers:
                reg_list = [f"R{i}" for i in range(self.env.registers)]
                if self.env.append_registers:
                    x = [xi + reg_list for xi in x]
                else:
                    x = [reg_list + xi for xi in x]
        else:
            if not self.full_sentence:
                #prepend y to x and make masks
                words, coefs = zip(*elements)
                x0 = [coef + word for coef, word in zip(coefs, words)]
            else:
                # read the full sentence and make masks
                x0 = elements
            y = x0
            x = [self.env.make_masks(xval, self.train, self.env.idx_to_infix([self.env.mask_index])) for xval in x0]
            nb_eqs = [0 for xi in x]
        #print(x,y)

        if self.decoder_only:
            xy = [ xi + ['<sep>'] + yi for xi, yi in zip(x,y)]
            xy = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in xy]
            xy, xy_len = self.batch_sequences(xy, self.env.pad_index, self.env.eos_index, self.env.eos_index)
            if self.train:
                return (xy, xy_len), torch.LongTensor(nb_eqs)
            else:
                x = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in x]
                x, x_len = self.batch_sequences(x, self.env.pad_index, self.env.eos_index, self.env.sep_index)
                return (x, x_len), (xy, xy_len), torch.LongTensor(nb_eqs)
        else:
            x = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in x]
            y = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in y]
            #use BOS for EOS in default enc-dec
            x, x_len = self.batch_sequences(x, self.env.pad_index, self.env.eos_index, self.env.eos_index)
            y, y_len = self.batch_sequences(y, self.env.pad_index, self.env.eos_index, self.env.eos_index)
            return (x, x_len), (y, y_len), torch.LongTensor(nb_eqs)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self.env, "rng"):
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.env.rng = np.random.RandomState(
                [worker_id, self.global_rank, self.env_base_seed]
            )
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{[worker_id, self.global_rank, self.env_base_seed]} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            self.env.rng = np.random.RandomState(None if self.type == "valid" else 0)

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.full_sentence:
            return self.read_sentence(index)
        else:
            if self.path is None:
                if self.type != "relations":
                    return self.generate_sample()
                else: raise ValueError
            else:
                return self.read_sample(index)
    def read_sample(self, index):
        """
        Read a sample.
        """
        idx = index
        if self.train:
            if self.batch_load:
                if index >= self.nextpos:
                    self.load_chunk()
                idx = index - self.basepos
            else:
                index = self.env.rng.randint(len(self.data))
                idx = index
        x, y = self.data[idx]
        if not self.type == "relations":
            x = x.split()
            y = y.split()
            assert len(x) >= 1 and len(y) >= 1
        else:
            x = list(x)
            assert len(x) >= 1 and  y is not None

        if self.numeric_import_input:
            res =[]
            for u in x:
                res.extend(self.env.input_encoder.encode(int(u)))
            x = res

        if self.numeric_import_output:
            val = int(y)
            y = self.env.output_encoder.encode(val)
        elif self.type == "relations":
            val = y
            if self.env.modulus > 1:
                val = val % self.env.modulus
            y = self.env.output_encoder.encode(val)
        elif self.prime_encoding == True:
            if self.env.modulus > 1: raise ValueError
            #no support for modular and prime simultaneously
            val = self.env.output_encoder.decode(y)
            y = self.env.output_encoder.encode(val)
        else:
            sign = 1 if y[0] == '+' else -1
            val = int(y[1])
            for i in range(2,len(y)):
                val = val * 1000 + int(y[i])
            if self.env.modulus <= 1:
                val = val * sign
            else:
                val = val % self.env.modulus
            y = self.env.output_encoder.encode(val)
        x=self.env.word_encoder.encode(x)
        #print(x)
        return x, y

    def read_sentence(self, index):
        """
        Read a sample that is not in the x->y pairs form (i.e., a 'sentence').
        """
        idx = index
        if self.train:
            if self.batch_load:
                if index >= self.nextpos:
                    self.load_chunk()
                idx = index - self.basepos
            else:
                index = self.env.rng.randint(len(self.data))
                idx = index
        instance= self.data[idx]
        sentence=[]
        #print(instance)
        for word,y in instance.items():
            #print(word,y)
            val = y[0]
            if self.env.modulus > 1:
                val = val % self.env.modulus
            elif self.prime_encoding == True:
                if self.env.modulus > 1: raise ValueError
                # no support for modular and prime simultaneously
                y = self.env.output_encoder.encode(val)
            else:
                y = self.env.output_encoder.encode(val)
            x=self.env.word_encoder.encode(word)
            sentence += y
            sentence += x
        #print(x)
        assert len(sentence) >= 1
        #print(sentence)
        return sentence

    def generate_sample(self):
        """
        Generate a sample.
        """
        while True:
            try:
                xy = self.env.gen_expr(self.type, self.task)
                if xy is None:
                    continue
                x, y = xy
                break
            except Exception as e:
                logger.error(
                    'An unknown exception of type {0} occurred for worker {4} in line {1} for expression "{2}". Arguments:{3!r}.'.format(
                        type(e).__name__,
                        sys.exc_info()[-1].tb_lineno,
                        "F",
                        e.args,
                        self.get_worker_id(),
                    )
                )
                continue
        self.count += 1

        return x, y
