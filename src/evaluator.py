# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
from logging import getLogger
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
from rels_utils import *
import csv
import torch
import numpy as np
import torchmetrics
from .utils import to_cuda,get_y_and_mask,inputs_embs_outputs_to_tsv,tokens_embs_to_tsv,plot_attn_maps
TOLERANCE_THRESHOLD = 1e-1
logger = getLogger()
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
from io import BytesIO
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


class Evaluator(object):

    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env
                
        if self.params.architecture != "decoder_only":
            self.encoder = (
                self.modules["encoder"].module
                if self.params.multi_gpu
                else self.modules["encoder"]
            )
            self.encoder.eval()

        if self.params.architecture != "encoder_only":
            self.decoder = (
                self.modules["decoder"].module
                if self.params.multi_gpu
                else self.modules["decoder"]
            )
            self.decoder.eval()

        self.counter = 0
    
    def run_all_evals(self):
        """
        Run all evaluations.

        """
        params = self.params
        scores = OrderedDict({"epoch": self.trainer.epoch})

        # save statistics about generated data
        if params.export_data:
            scores["total"] = self.trainer.total_samples
            return scores

        if params.eval_relations and (params.relations_path is not None):
            data_types = ["valid","relations"]
        else: data_types = ["valid"]

        with torch.no_grad():
            for data_type in data_types:
                for task in params.tasks:
                    self.enc_dec_step(data_type, task, scores)

        return scores

    def display_logs(self, logs, offset, eval_path, data_type):  # FC A revoir
        """
        Display detailed results about success / fails.
        """

        if self.params.eval_verbose == 0 and data_type != "relations":
            return

        eval_dict = {}
        for i, res in sorted(logs.items()):
            n_valid = sum([int(v) for *_, v in res["hyps"]])
            s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\n"
            s += f"src={res['src']}\ntgt={res['tgt']}\n"
            for hyp, *_, valid in res["hyps"]:
                # if score is None:
                s += f"{int(valid)} {hyp}\n"
                # else:
                #    s += f"{int(valid)} {score :.3e} {hyp}\n"
                if (not self.params.beam_eval) and ("boots" in self.params.tasks) and not ("mask" in self.params.operation):
                    try:
                        eval_dict[res['src']] = int(hyp)
                    except:
                        print(f"Error: {hyp} is not an int")
                        eval_dict[res['src']] = None
            if data_type != "relations" and (self.params.eval_verbose != 0) and not (self.params.eval_verbose_dict):
                f_export = open(eval_path, "a")
                if self.params.eval_verbose_print:
                    logger.info(s)
                f_export.write(s + "\n")
                f_export.close()

        if data_type == "relations" or (self.params.eval_verbose != 0 and (self.params.eval_verbose_dict)):
            try:
                with open(eval_path, "r") as jsonFile:
                    data = json.load(jsonFile)
            except FileNotFoundError:
                data = {}
                pass

            data.update(eval_dict)
            with open(eval_path, "w") as jsonFile:
                json.dump(data, jsonFile)

        return

    def enc_dec_step(self, data_type, task, scores):
        """
        Encoding / decoding step.
        """
        params = self.params
        env = self.env
        max_beam_length = params.max_output_len + 2

        if params.beam_eval:
            assert params.eval_verbose in [0, 1, 2]
        else:
            assert params.eval_verbose in [0, 1]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in env.TRAINING_TASKS
        iterators = []
        paths=[]
        # iterator
        if data_type == "relations":
            file_path = os.path.join(params.relations_path, "rel_instances_all_relations.json")
            print(file_path)
            iterator = self.env.create_test_iterator(
                data_type,
                task,
                data_path=file_path,
                batch_size=params.batch_size_eval,
                params=params,
                size=params.eval_size,
            )
            iterators.append(iterator)
            paths.append(file_path)
        else:
            iterator = self.env.create_test_iterator(
                data_type,
                task,
                data_path=self.trainer.data_path,
                batch_size=params.batch_size_eval,
                params=params,
                size=params.eval_size,
            )
            iterators.append(iterator)
            paths.append(self.trainer.data_path)

        for rel_path, iterator in zip(paths,iterators):
            eval_size = len(iterator.dataset)
            # evaluation details
            if data_type=="relations":
                key = "all_relations"
                print(rel_path)
                print(key)
            else:
                key = data_type
            if params.eval_verbose or data_type=="relations":
                if params.beam_eval:
                    eval_path = os.path.join(
                        params.dump_path, f"eval.beam.{key}.{task}.{scores['epoch']}"
                    )
                else:
                    eval_path = os.path.join(
                        params.dump_path, f"eval.{key}.{task}.{scores['epoch']}"
                    )
                logger.info(f"Writing evaluation results in {eval_path} ...")


            # stats
            xe_loss = 0

            #these should be the same no matter the task, I think? Valid is valid
            n_valid = torch.zeros(10000, dtype=torch.long) #correct is in the beam AND it's within float tolerance
            n_total = torch.zeros(10000, dtype=torch.long)
            n_valid_additional = np.zeros(1 + len(env.additional_tolerance), dtype=int) #try other float tolerances
            n_perfect_match = 0 #correct the first time, no need for beam
            n_phrases_perfect = 0 #for masking
            n_correct = 0 #correct is in the beam, but maybe not within float tolerance

            n_eval_metrics={}
            for metric in env.hyp_eval_metrics:
                n_eval_metrics[metric]=0

            # initialize metric
            if params.cpu==False:
                tokenwise_acc_macro = torchmetrics.classification.Accuracy(task="multiclass", num_classes=len(env.words),average="macro").cuda()
                tokenwise_acc_micro = torchmetrics.classification.Accuracy(task="multiclass", num_classes=len(env.words)).cuda()
            else:
                tokenwise_acc_macro = torchmetrics.classification.Accuracy(task="multiclass", num_classes=len(env.words),average="macro")
                tokenwise_acc_micro = torchmetrics.classification.Accuracy(task="multiclass", num_classes=len(env.words))

            for (x1, len1), (x2, len2), nb_ops in iterator:
                #y, pred_mask = get_y_and_mask(x2, len2)
                # cuda
                x1_, len1_, x2, len2 = to_cuda(x1, len1, x2, len2)
                # target words to predict
                if params.architecture == "encoder_decoder":
                    alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
                    pred_mask = (
                        alen[:, None] < len2[None] - 1
                    )  # do not predict anything given the last target word
                    y = x2[1:].masked_select(pred_mask[:-1])
                    assert len(y) == (len2 - 1).sum().item()
                elif params.architecture == "encoder_only":
                    alen = torch.arange(len1_.max(), dtype=torch.long, device=len2.device)
                    torch.set_printoptions(profile="full")
                    pred_mask = (
                                    (alen[:, None] < len2[None]) & (alen[:, None] > torch.zeros_like(len2)[None])
                    )
                    y= torch.cat((x2,torch.full((len1_.max()-len2.max(),len2.size(0)),self.env.eos_index,device=len2.device)),0)
                    y = y.masked_select(pred_mask)
                elif params.architecture == "decoder_only":
                    #pred mask should ignore word and only check coef
                    alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
                    pred_mask = (
                        (alen[:, None] < len2[None] - 1) #mask out last element
                        & (alen[:, None] > len1_[None] - 2) #mask out word and sep token
                    )
                    # do not predict anything given the last target word
                    y = x2[1:].masked_select(pred_mask[:-1])
                    #print(y,x2,pred_mask)
                    assert len(y) == (len2 - len1_).sum().item()

                bs = len(len1)
                if params.architecture == "encoder_decoder":
                    if self.params.eval_only and self.params.export_encoding:
                        embs, encoded = self.encoder("fwd", x=x1_, lengths=len1_, causal=False)
                        inputs_embs_outputs_to_tsv(self.env, x1_, x2, encoded,
                                                f"./runs_eval/{self.params.exp_id}_{key}_encodings.tsv",
                                                f"./runs_eval/{self.params.exp_id}_{key}_metadata.tsv",(data_type=="relations"))
                        if data_type != "relations":
                            tokens_embs_to_tsv(self.env, x1_, embs,
                                                    f"./runs_eval/{self.params.exp_id}_token_embs.tsv",
                                                    f"./runs_eval/{self.params.exp_id}_token_metadata.tsv")
                        out_embs, decoded = self.decoder(
                            "fwd",
                            x=x2,
                            lengths=len2,
                            causal=True,
                            src_enc=encoded.transpose(0, 1),
                            src_len=len1_,
                        )
                        word_scores, loss = self.decoder(
                            "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
                        )
                        tokens_embs_to_tsv(self.env, x2, out_embs,
                                           f"./runs_eval/{self.params.exp_id}_out_token_embs.tsv",
                                           f"./runs_eval/{self.params.exp_id}_out_token_metadata.tsv")

                    else:
                        encoded = self.encoder("fwd", x=x1_, lengths=len1_, causal=False)
                        decoded = self.decoder(
                            "fwd",
                            x=x2,
                            lengths=len2,
                            causal=True,
                            src_enc=encoded.transpose(0, 1),
                            src_len=len1_,
                        )
                        word_scores, loss = self.decoder(
                            "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
                        )
                elif params.architecture == "encoder_only":
                    encoded = self.encoder("fwd", x=x1_, lengths=len1_, causal=False)
                    word_scores, loss = self.encoder(
                        "predict", tensor=encoded, pred_mask=pred_mask, y=y, get_scores=True
                    )
                elif params.architecture == "decoder_only":
                    decoded = self.decoder("fwd", x=x2, lengths=len2, causal=True, src_enc=None, src_len=None)
                    word_scores, loss = self.decoder(
                        "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
                    )

                if data_type == "valid" and params.plot_attn_maps and not params.beam_eval and params.eval_verbose < 2:
                    # upload the teacher-forced attn maps, since they're all correct
                    enc_attn_scores = self.encoder.all_scores
                    dec_attn_scores = self.decoder.all_scores

                # correct outputs per sequence / valid top-1 predictions
                t = torch.zeros_like(pred_mask, device=y.device)
                top_preds = word_scores.max(1)[1]
                t[pred_mask] += word_scores.max(1)[1] == y

                if params.architecture == "decoder_only":
                    valid = (t.sum(0) == len2 - len1_).cpu().long()
                else:
                    valid = (t.sum(0) == len2 - 1).cpu().long()
                n_perfect_match += valid.sum().item()

                # export evaluation details
                beam_log = {}
                # stats
                xe_loss += loss.item() * len(y)
                n_valid.index_add_(-1, nb_ops, valid)
                n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

                for i in range(len(len1)):
                    if params.architecture == "decoder_only":
                        src = env.idx_to_infix(x2[1 : len1[i] - 1, i].tolist(), True)
                        tgt = env.idx_to_infix(x2[len1[i] : len2[i] - 1, i].tolist(), False)
                    else:
                        src = env.idx_to_infix(x1[1 : len1[i] - 1, i].tolist(), True)
                        tgt = env.idx_to_infix(x2[1 : len2[i] - 1, i].tolist(), False)

                    #log it if the teacher-forced prediction is good
                    if valid[i]:
                        if ("mask" in self.params.operation):
                            n_phrases_perfect += len(env.list_to_keyvals(tgt).keys())

                        beam_log[i] = {"src": src, "tgt": tgt, "hyps": [(tgt,) + ((True,) * (len(env.hyp_eval_metrics)+1))]}
                        if params.eval_verbose < 2 and not params.beam_eval:

                            # if eval_verbose < 2, the decoding step will skip correct gens,
                            # so tokenwise acc will not be logged there: so need to do it here for these
                            tgt_tensor = x2[1: len2[i] - 1, i]
                            mac_acc = tokenwise_acc_macro(tgt_tensor, tgt_tensor)
                            mic_acc = tokenwise_acc_micro(tgt_tensor, tgt_tensor)

                            #plot attn
                            if (self.counter < self.params.examples_to_plot and data_type == "valid"
                                and params.plot_attn_maps and not params.beam_eval and params.eval_verbose < 2):
                                self.counter=plot_attn_maps(self.trainer, self.counter, enc_attn_scores[i], dec_attn_scores[i],
                                                    x1[:len1[i], i].tolist(),
                                                    x2[:len2[i], i].tolist(), "correct")

                # invalid top-1 predictions - check if there is a solution in the beam
                invalid_idx = (1 - valid).nonzero().view(-1)
                logger.info(
                    f"({n_total.sum().item()}/{eval_size}) Found "
                    f"{bs - len(invalid_idx)}/{bs} valid top-1 predictions. "
                    f"Generating solutions ..."
                )

                # continue if the whole batch is correct. if eval_verbose = 2, perform
                # a full beam search, even on correct greedy generations
                if valid.sum() == len(valid) and (params.eval_verbose == 1 or data_type=="relations"):
                    self.display_logs(beam_log, offset=n_total.sum().item() - bs,eval_path=eval_path, data_type=data_type)
                    continue

                # generate
                if params.beam_eval:
                    _, _, generated = self.decoder.generate_beam(
                        encoded.transpose(0, 1),
                        len1_,
                        beam_size=params.beam_size,
                        length_penalty=params.beam_length_penalty,
                        early_stopping=params.beam_early_stopping,
                        max_len=max_beam_length,
                    )
                else:
                    if params.architecture == "encoder_decoder":
                        generated, _ = self.decoder.generate(
                            encoded.transpose(0, 1),
                            len1_,
                            max_len=max_beam_length,
                        )
                        generated=generated.transpose(0, 1)
                    elif params.architecture == "encoder_only":
                        generated = self.encoder.decode(x1_, len1_, max_beam_length)
                    else:
                        #for dec-only, only generate up to the "true" max len
                        max_beam_length = len(pred_mask)
                        generated, _ = self.decoder.generate(x1_, len1_, max_beam_length)
                        generated=generated.transpose(0, 1)

                    if data_type == "valid" and params.plot_attn_maps and params.eval_verbose < 2:
                        dec_attn_scores = self.decoder.all_scores

                # prepare inputs / hypotheses to check
                # if eval_verbose < 2, no beam search on equations solved greedily
                inputs = []
                for i in range(len(generated)):
                    if valid[i] and params.eval_verbose < 2:
                        continue
                    if params.beam_eval:
                        for j, (score, hyp) in enumerate(sorted(generated[i].hyp, key=lambda x: x[0], reverse=True)):
                            inputs.append(
                                {
                                    "i": i,
                                    "j": j,
                                    "score": score,
                                    "src": x1[1: len1[i] - 1, i].tolist(),
                                    "tgt": x2[1: len2[i] - 1, i].tolist(),
                                    "hyp": hyp[1:].tolist(),
                                    "task": task,
                                }
                            )
                    else:
                        if params.architecture == "encoder_only":
                            try:
                                end_idx = generated[i][1:].tolist().index(env.eos_index) + 1
                            except ValueError:
                                #if enc-only fails to generate a stop token, we don't append it at max_len like
                                #enc-dec, so patch it in here
                                end_idx = (len(generated[i])-1)
                        else:
                            end_idx = generated[i][1:].tolist().index(env.eos_index)+1
                        if params.architecture == "decoder_only":
                            inputs.append(
                                {
                                    "i": i,
                                    "src": x2[1 : len1[i] - 1, i].tolist(),
                                    "tgt": x2[len1[i] : len2[i] - 1, i].tolist(),
                                    "hyp": generated[i][len1[i]:end_idx].tolist(),
                                    "task": task,
                                }
                            )
                        else:
                            inputs.append(
                            {
                                "i": i,
                                "src": x1[1 : len1[i] - 1, i].tolist(),
                                "tgt": x2[1 : len2[i] - 1, i].tolist(),
                                "hyp": generated[i][1:end_idx].tolist(),
                                "task": task,
                            }
                        )
                        #calculate tokenwise macro and micro accuracy. Pad with nulls
                        hyp_tensor = generated[i][1:end_idx]
                        tgt_tensor = x2[1 : len2[i] - 1, i]
                        seq= pad_sequence([hyp_tensor, tgt_tensor], batch_first=True)
                        mac_acc = tokenwise_acc_macro(seq[0],seq[1])
                        mic_acc = tokenwise_acc_micro(seq[0],seq[1])

                        # plot attn
                        if (self.counter < self.params.examples_to_plot and data_type == "valid"
                                and params.plot_attn_maps and not params.beam_eval and params.eval_verbose < 2):
                            self.counter=self.plot_attn_maps(self.trainer,self.counter,enc_attn_scores[i],
                                                             dec_attn_scores[i],x1[:len1[i], i].tolist(),
                                                             x2[:len2[i], i].tolist(), "incorrect")
                # check hypotheses with multiprocessing
                outputs = []
                if params.windows is True:
                    for inp in inputs:
                        if "mask" in self.env.operation:
                            outputs.append(env.check_masked_hypothesis(inp))
                        else:
                            outputs.append(env.check_hypothesis(inp))
                else:
                    with ProcessPoolExecutor(max_workers=20) as executor:
                        if "mask" in self.env.operation:
                            for output in executor.map(env.check_masked_hypothesis, inputs, chunksize=1):
                                outputs.append(output)
                        else:
                            for output in executor.map(env.check_hypothesis, inputs, chunksize=1):
                                outputs.append(output)
                # logger.info(f"{len(inputs)} input, {len(outputs)} output processed")
                # read results
                for i in range(bs):
                    # select hypotheses associated to current equation
                    if params.beam_eval: gens = sorted([o for o in outputs if o["i"] == i], key=lambda x: x["j"])
                    else: gens = [o for o in outputs if o["i"] == i]
                    assert (len(gens) == 0) == (valid[i] and params.eval_verbose < 2)
                    assert (i in beam_log) == valid[i]
                    if len(gens) == 0:
                        continue
                    if not params.beam_eval: assert len(gens) == 1

                    # source / target
                    gen = gens[0]
                    # logger.info(f"gen: {gen}")
                    src = gen["src"]
                    tgt = gen["tgt"]
                    beam_log[i] = {"src": src, "tgt": tgt, "hyps": []}
                    hyp_eval_counts = dict.fromkeys(n_eval_metrics, 0)
                    curr_additional = np.zeros(1 + len(env.additional_tolerance), dtype=int)
                    curr_correct = 0
                    curr_valid = 0

                    # for each hypothesis
                    for j, gen in enumerate(gens):
                        # sanity check
                        assert (
                                gen["src"] == src
                                and gen["tgt"] == tgt
                                and gen["i"] == i
                        )
                        # sanity check
                        if params.beam_eval:
                            assert (gen["j"] == j)

                        # if hypothesis is correct, and we did not find a correct one before
                        is_valid = gen["is_valid"]
                        is_b_valid = is_valid >= 0.0 and is_valid < env.float_tolerance

                        #if we haven't found a valid one yet
                        if not valid[i]:
                            if ("mask" in self.params.operation):
                                for metric in n_eval_metrics:
                                    if gen["hyp_evals"][metric] >= 0:
                                        hyp_eval_counts[metric] = gen["hyp_evals"][metric]
                            else:
                                #log additional metrics
                                for metric in n_eval_metrics:
                                    if gen["hyp_evals"][metric] >= 0:
                                        hyp_eval_counts[metric] = 1

                            #if it's valid now, get the valid metrics. Should rework this
                            if is_valid >= 0.0:
                                curr_correct = 1
                                if is_valid < env.float_tolerance:
                                    curr_valid = 1
                                for k, tol in enumerate(env.additional_tolerance):
                                    if is_valid < tol:
                                        curr_additional[k] = 1

                        # update beam log
                        my_beam_out = [gen["hyp"]]
                        for metric in gen["hyp_evals"]:
                            my_beam_out.append(hyp_eval_counts[metric] > 0)

                        my_beam_out.append(is_b_valid)
                        beam_log[i]["hyps"].append(tuple(my_beam_out))

                        if not valid[i]:
                            n_correct += curr_correct
                            for metric in env.hyp_eval_metrics:
                                n_eval_metrics[metric] += hyp_eval_counts[metric]

                            for k, tol in enumerate(env.additional_tolerance):
                                n_valid_additional[k] += curr_additional[k]
                            valid[i] = curr_valid
                            n_valid[nb_ops[i]] += curr_valid

                # valid solutions found with beam search
                logger.info(
                    f"    Found {valid.sum().item()}/{bs} solutions in beam hypotheses. "
                )

                # export evaluation details
                if params.eval_verbose or data_type=="relations":
                    assert len(beam_log) == bs
                    self.display_logs(beam_log, offset=n_total.sum().item() - bs, eval_path=eval_path,data_type=data_type)

            # evaluation details
            if params.eval_verbose or data_type=="relations":
                logger.info(f"Evaluation results written in {eval_path}")


            # log
            _n_valid = n_valid.sum().item()
            _n_total = n_total.sum().item()
            logger.info(
                f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) "
                f"equations were evaluated correctly."
            )
            assert _n_total == eval_size

            if (data_type != "relations") or (data_type == "relations" and "all_relations" in eval_path):
                # compute perplexity and prediction accuracy
                scores[f"{data_type}_{task}_xe_loss"] = xe_loss / _n_total
                scores[f"{data_type}_{task}_acc"] = 100.0 * _n_valid / _n_total
                scores[f"{data_type}_{task}_perfect"] = 100.0 * n_perfect_match / _n_total
                scores[f"{data_type}_{task}_correct"] = (
                    100.0 * (n_perfect_match + n_correct) / _n_total
                )

            if data_type != "relations":
                # metric on all batches using custom accumulation.
                # Should default to zero if beam_eval is on, since not well-defined with beam
                mic_acc = tokenwise_acc_micro.compute()
                mac_acc = tokenwise_acc_macro.compute()

                scores[f"{data_type}_{task}_tokenwise_macro_acc"] = mac_acc.item()
                scores[f"{data_type}_{task}_tokenwise_micro_acc"] = mic_acc.item()

            if "mask" in self.params.operation:
                for metric in n_eval_metrics:
                    if "phrases" in metric: continue
                    else:
                        scores[f"{data_type}_{task}_acc_{metric}_phrases"] = (
                            100.0 * (n_phrases_perfect + n_eval_metrics[metric]) / (n_eval_metrics["nphrases"] + n_phrases_perfect)
                        )
            else:
                for metric in n_eval_metrics:
                    scores[f"{data_type}_{task}_acc_{metric}"] = (
                        100.0 * (n_perfect_match + n_eval_metrics[metric]) / _n_total
                    )

            for i in range(len(env.additional_tolerance)):
                scores[f"{data_type}_{task}_additional_{i+1}"] = (
                    100.0 * (n_perfect_match + n_valid_additional[i]) / _n_total
                )

                # per class perplexity and prediction accuracy
                for i in range(len(n_total)):
                    if n_total[i].item() == 0:
                        continue
                    e = env.decode_class(i)
                    scores[f"coef_class/{data_type}_{task}_acc_{e}"] = (
                            100.0 * n_valid[i].item() / max(n_total[i].item(), 1)
                    )
                    if n_valid[i].item() > 0:
                        logger.info(
                            f"{e}: {n_valid[i].item()} / {n_total[i].item()} "
                            f"({100. * n_valid[i].item() / max(n_total[i].item(), 1)}%)"
                        )

            tokenwise_acc_micro.reset()
            tokenwise_acc_macro.reset()

            rel_acc = {}
            rel_acc_allcorr = {}
            rel_acc_magcorr = {}
            rel_acc_signcorr = {}
            rel_acc_magcorr_norelreq = {}
            rel_acc_signcorr_norelreq = {}

            #get the model output for relation
            if params.eval_relations and (params.relations_path is not None) and data_type == "relations":
                with open(eval_path, 'r') as symbfile:
                    raw_evals = json.load(symbfile)
                    if params.hardcode_trivial_zeroes:
                        raw_evals = replace_trivial0_terms(raw_evals, return_symb=True)
                    for relation in os.listdir(params.relations_path):
                        file_path = os.path.join(params.relations_path, relation)
                        with open(file_path, 'r') as openfile:
                            truth_symb = json.load(openfile)
                            rel_instance_eval_symb = update_rel_instances_in_symb(truth_symb, raw_evals)
                            rel_key = os.path.split(file_path)[-1].split(".")[0].replace("rel_instances_", "")
                            percent = check_rel(rel_instance_eval_symb, return_rel_info=False,
                                                p_norm=None)
                            scores[f"relations/{rel_key}_{task}_rel_acc"] = percent
                            rel_acc.update({rel_key: [percent, len(rel_instance_eval_symb)]})

                            percent_corr, percent_mag_corr, percent_sign_corr = check_coeffs_in_rel(rel_instance_eval_symb,truth_symb)
                            scores[f"rel_correct/{rel_key}_{task}_coef_correct"] = percent_corr
                            scores[f"rel_mag_correct/{rel_key}_{task}_mag_correct"] = percent_mag_corr
                            scores[f"rel_sign_correct/{rel_key}_{task}_sign_correct"] = percent_sign_corr

                            percent_corr_norel, percent_mag_corr_norel, percent_sign_corr_norel = check_coeffs_in_rel(rel_instance_eval_symb,truth_symb,
                                                                                                                      return_counts = False, require_satisfied = False)
                            #scores[f"rel_correct/{rel_key}_{task}_coef_correct"] = percent_corr
                            scores[f"rel_mag_correct_norelreq/{rel_key}_{task}_mag_correct_norelreq"] = percent_mag_corr_norel
                            scores[f"rel_sign_correct_norelreq/{rel_key}_{task}_sign_correct_norelreq"] = percent_sign_corr_norel

                            rel_acc_allcorr.update({key: [percent_corr, len(rel_instance_eval_symb)]})
                            rel_acc_magcorr.update({key: [percent_mag_corr, len(rel_instance_eval_symb)]})
                            rel_acc_signcorr.update({key: [percent_sign_corr, len(rel_instance_eval_symb)]})
                            rel_acc_magcorr_norelreq.update({key: [percent_mag_corr_norel, len(rel_instance_eval_symb)]})
                            rel_acc_signcorr_norelreq.update({key: [percent_sign_corr_norel, len(rel_instance_eval_symb)]})

                        """
                        percent_normed = check_rel(rel_instance_in_symb_list, return_rel_sum=False,
                                            p_norm=scores[f"valid_{task}_acc"]/100)
                        scores[f"normed_relations/{key}_{task}_rel_acc_pnormed"] = percent_normed
                        rel_acc_pnorm.update({key: [percent_normed, len(rel_instance_in_symb_list)]})
                        """