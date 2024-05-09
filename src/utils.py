# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import sys
import math
import time
import pickle
import random
import getpass
import argparse
import subprocess
import torch
import csv
import pandas as pd
import errno
import signal
from PIL import Image
from io import BytesIO
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
import wandb
import json
from scipy.spatial.distance import cdist
import numpy as np
from functools import wraps, partial, reduce
from .logger import create_logger

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

#HTCondor runs as "nobody", which makes getuser fail
try:
    DUMP_PATH = './checkpoint/%s/dumped' % getpass.getuser()
except:
    DUMP_PATH = './checkpoint/htcondor/dumped'

CUDA = True


def is_power_of_p(n,p):
    if p == 0: raise ValueError
    if n == 0: return False
    else: logp = (math.log10(n) / math.log10(p))
    return (math.ceil(logp) == math.floor(logp))

def biggest_power_of_p(n, p):
    n = abs(n)
    biggest = 0
    if n == 0:
        return biggest
    else:
        try:
            factors = set(reduce(list.__add__,
                    ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
            for factor in factors:
                if (is_power_of_p(factor, p)) and factor > biggest: biggest = factor
        except:
            print(n,p)
            biggest = -999999999999
    return biggest

def encode_num(number, base, mod=0):
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

def padic_order_equals(v,w,p):
    return biggest_power_of_p(v, p) == biggest_power_of_p(w, p)

def filter_output_between_strings(input: [str], separator: str):
    separator_indexes = [index for index, word in enumerate(input) if word == separator]
    for example_index, (start, end) in enumerate(zip(separator_indexes[:-1], separator_indexes[1:]), start=1):
        return input[start + 1:end]

def inputs_embs_outputs_to_tsv(env, inputs, outputs, embs, data_file, metadata_file,is_relations):
    file = open(data_file, "a")
    meta_file = open(metadata_file, "a")
    writer = csv.writer(file, delimiter='\t')
    meta_writer = csv.writer(meta_file, delimiter='\t')
    if os.path.getsize(metadata_file) == 0:
        meta_header = ["word","coef"]
        meta_writer.writerow(meta_header)
    inputs = torch.transpose(inputs,0,1).cpu().numpy()
    outputs = torch.transpose(outputs, 0, 1).cpu().numpy()
    embs = torch.transpose(embs, 0, 1).cpu().numpy()
    for input, emb, output in zip(inputs, embs, outputs):
        in_word = env.idx_to_infix(input, True)[3:-3]
        if is_relations:
            out_word = filter_output_between_strings(env.idx_to_infix(output, False),"<s>")
        else:
            out_word = filter_output_between_strings(env.idx_to_infix(output, False),"<s>")
        my_emb = emb.flatten().tolist()
        meta_writer.writerow([in_word, out_word])
        writer.writerow(my_emb)
    file.close()
    meta_file.close()

def tokens_embs_to_tsv(env, inputs, embs, data_file, metadata_file):
    if os.path.isfile(data_file):
        #read the files and put it all into a df
        embs_df = pd.read_csv(data_file, sep='\t')
    else:
        embs_df = pd.DataFrame()

    if os.path.isfile(metadata_file):
        meta_df = pd.read_csv(metadata_file, sep='\t')
    else:
        meta_df = pd.DataFrame()
    # if we already have a,b,c,d,e,f,<bos>,<eos>, we're done
    if env.operation == "coeffs" and len(meta_df) > 6: return
    full_df = pd.concat([meta_df, embs_df], axis=1, ignore_index=True)
    all_inputs = torch.transpose(inputs,0,1).cpu().numpy()
    #transpose it later
    all_embs = embs.cpu().numpy()
    for inputs, embs in zip(all_inputs, all_embs):
        in_tokens = [env.id2word[wid] for wid in inputs]
        for input, emb in zip(in_tokens, embs):
            emb_df = pd.DataFrame(emb).T
            input_df=pd.DataFrame({0:[input]})
            my_df = pd.concat([input_df, emb_df], axis=1, ignore_index=True)
            full_df = pd.concat([full_df, my_df], axis=0, ignore_index=True)
        full_df=full_df.drop_duplicates()
    #write the dfs to tsvs
    full_df.to_csv(metadata_file, index=False, sep='\t', columns=[0],header=False)
    out_embs = full_df.drop(0,axis=1)
    out_embs.to_csv(data_file, index=False, sep='\t',header=False)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))
    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0



    # create a logger and a tb writer
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")

    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    params.dump_path = DUMP_PATH if params.dump_path == '' else params.dump_path
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == '':
        chronos_job_id = os.environ.get('CHRONOS_JOB_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    if not CUDA:
        return args
    return [None if x is None else x.cuda() for x in args]


class TimeoutError(BaseException):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):

    def decorator(func):

        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator

def is_all_zeroes(s):
    for i in s:
        if (i != '0') and (i != '+') and (i != '-'):
            return False
    return True

def get_y_and_mask(x2,len2):
    # target words to predict
    alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
    pred_mask = (
            alen[:, None] < len2[None] - 1
    )  # do not predict anything given the last target word
    y = x2[1:].masked_select(pred_mask[:-1])
    assert len(y) == (len2 - 1).sum().item()
    return y, pred_mask

def get_nearest_points_in_df(point_df, points_df,k):
    inds = np.argpartition(cdist([point_df], points_df,metric='cosine'),k)[0][:k]
    return [points_df.iloc[ind].name for ind in inds]

def evaluate_relation_embs(rel_name,run_id,rel_path,embs_path,max_to_show):
    #open the json with the relation in it
    with open(f'{rel_path}/rel_instances_{rel_name}.json') as json_data:
        rel_dicts = json.load(json_data)

    metadata = pd.read_csv(f'{embs_path}/{run_id}_{rel_name}_metadata.tsv', sep='\t')
    encodings = pd.read_csv(f'{embs_path}/{run_id}_{rel_name}_encodings.tsv', sep='\t', header=None)

    fulldf_metadata = pd.read_csv(f'{embs_path}/{run_id}_valid_metadata.tsv', sep='\t')
    fulldf_encodings = pd.read_csv(f'{embs_path}/{run_id}_valid_encodings.tsv', sep='\t', header=None)
    df = metadata.join(encodings, how='outer')
    df=df.drop_duplicates()
    fulldf = fulldf_metadata.join(fulldf_encodings, how='outer')
    fulldf=fulldf.drop_duplicates()
    mega_df=pd.concat([df,fulldf])
    mega_df=mega_df.drop_duplicates()
    mega_df=mega_df.set_index(["word"])
    mega_df=mega_df.drop(columns=["coef"])
    mega_df = mega_df[~mega_df.index.duplicated(keep='first')]
    df=df.set_index(["word"])
    df=df.drop(columns=["coef"])
    df = df[~df.index.duplicated(keep='first')]

    fulldf_withcoeffs=fulldf.set_index(["word","coef"])
    fulldf=fulldf.set_index(["word"])
    fulldf=fulldf.drop(columns=["coef"])
    fulldf = fulldf[~fulldf.index.duplicated(keep='first')]
    for ind,instance in enumerate(rel_dicts):
        print(instance)
        vecsum = np.zeros(256)
        mywords=[]
        for word_ind,(word,coeff) in enumerate(instance.items()):
            if word_ind == 0:
                print(f"word={word},coef={coeff[0]}")
                continue
            mywords+=[word]
            myvec=df.loc[word].apply(lambda x: x*1).to_numpy()
            vecsum=vecsum + myvec
        nearests=get_nearest_points_in_df(pd.Series(vecsum),mega_df,len(instance.keys()))
        mynearests = [item for item in nearests if item not in mywords]
        print(f"nearest={mynearests}")
        print("")
        if ind > max_to_show:
            return

def plot_attn_maps(trainer, counter, enc_attn_scores,dec_attn_scores,src_label,tgt_label,tag,log_to_wandb=True):
    src_label = trainer.env.idx_to_infix(src_label, False)
    tgt_label = trainer.env.idx_to_infix(tgt_label, False)
    my_cmap = cm.coolwarm
    for l, layerscore in enumerate(enc_attn_scores):
        layerscore = layerscore.cpu().squeeze()
        my_min = 100000000
        my_max = -100000000
        for headscore in layerscore:
            if torch.max(headscore) > my_max: my_max = torch.max(headscore)
            if torch.min(headscore) < my_min: my_min = torch.min(headscore)
        for h, headscore in enumerate(layerscore):
            fig, ax = plt.subplots(figsize=(4, 4))
            my_show = ax.matshow(headscore, cmap=my_cmap, vmin=my_min, vmax=my_max)
            ax.set_xticks(np.arange(len(src_label)),labels=src_label)
            ax.set_yticks(np.arange(len(src_label)), labels=src_label)
            # We change the fontsize of minor ticks label
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            fig.colorbar(my_show, ax=ax)
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png')
            if not log_to_wandb:
                pimg = F.pil_to_tensor(Image.open(buf))
                trainer.writer.add_image(f"{tag}_{counter}/enc_attn_layer_{l}", pimg, global_step=l * 10 + h)
            else:
                pil = Image.open(buf)
                images = wandb.Image(
                    pil,
                    caption=f"{tag}_{counter}/enc_attn_layer_{l}/head{h}"
                )
                wandb.log({f"{tag}_{counter}/enc_attn_layer_{l}/head{h}": images}, step=trainer.epoch)

            plt.close()

    for l, layerscore in enumerate(dec_attn_scores):
        layerscore = layerscore.cpu().squeeze()
        fig = plt.figure(figsize=(4, 4))
        ax = plt.gca()
        num_heads = layerscore.shape[0]
        my_min = 100000000
        my_max = -100000000
        for headscore in layerscore:
            if torch.max(headscore) > my_max: my_max = torch.max(headscore)
            if torch.min(headscore) < my_min: my_min = torch.min(headscore)
        for h, headscore in enumerate(layerscore):
            ax = plt.subplot((num_heads * 100 + 10 + h + 1), sharex=ax)
            my_show = ax.matshow(headscore, cmap=my_cmap, vmin=my_min, vmax=my_max)
            ax.set_xticks(np.arange(len(src_label)),labels=src_label)
            ax.set_yticks(np.arange(len(tgt_label)), labels=tgt_label)
            # We change the fontsize of minor ticks label
            ax.tick_params(axis='both', which='major', labelsize=5, bottom=False)
            ax.tick_params(axis='both', which='minor', labelsize=5, bottom=False)
            if h != 0:
                ax.tick_params(axis='x',labeltop=False, length=0)
        cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        fig.colorbar(my_show, cax=cb_ax)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        pimg = F.pil_to_tensor(Image.open(buf))
        if not log_to_wandb:
            pimg = F.pil_to_tensor(Image.open(buf))
            trainer.writer.add_image(f"{tag}_{counter}/dec_attn_layer_{l}", pimg, global_step=l)
        else:
            pil = Image.open(buf)
            images = wandb.Image(
                pil,
                caption=f"{tag}_{counter}/dec_attn_layer_{l}/all_heads"
            )
            wandb.log({f"{tag}_{counter}/dec_attn_layer_{l}/all_heads": images}, step=trainer.epoch)
        plt.close()
    counter += 1
    return counter
def combine_jsons(rel_path):
    json_files = os.listdir(rel_path)
    try:
        json_files.remove("all_relations.json")
        print("removing old combined file")
    except:
        print("No existing combined file, creating")
    # Create an empty list to store the Python objects.
    python_objects = []
    # Load each JSON file into a Python object.
    for json_file in json_files:
        with open(rel_path + json_file, "r") as f:
            python_objects += json.load(f)

    # Dump all the Python objects into a single JSON file.
    with open(f"{rel_path}all_relations.json", "w") as f:
        json.dump(python_objects, f)
