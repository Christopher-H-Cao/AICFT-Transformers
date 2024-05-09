# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from collections import OrderedDict
from matplotlib import cm
from concurrent.futures import ProcessPoolExecutor
import os
import imgkit
import csv
from matplotlib.ticker import MultipleLocator
import torch
import torchvision.transforms.functional as F
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
from .utils import to_cuda,is_all_zeroes,get_y_and_mask
TOLERANCE_THRESHOLD = 1e-1
logger = getLogger()
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from captum.attr import visualization as viz
from io import BytesIO

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def inputs_embs_outputs_to_tsv(env, inputs, outputs, embs, data_file, metadata_file):
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
        in_word = idx_to_infix(env,input)[3:-3]
        out_word = idx_to_infix(env,output)[3:-3]
        my_emb = emb.flatten().tolist()
        meta_writer.writerow([in_word, out_word])
        writer.writerow(my_emb)
    file.close()
    meta_file.close()

def check_hypothesis(eq):
    """
    Check a hypothesis for a given equation and its solution.
    """
    env = Evaluator.ENV

    src = [env.id2word[wid] for wid in eq["src"]]
    tgt = [env.id2word[wid] for wid in eq["tgt"]]
    hyp = [env.id2word[wid] for wid in eq["hyp"]]

    # update hypothesis
    eq["src"] = env.input_to_infix(src)
    eq["tgt"] = env.output_to_infix(tgt)
    eq["hyp"] = env.output_to_infix(hyp)
    try:
        m, s1, s2, nb = env.check_prediction(src, tgt, hyp)
    except Exception:
        m = -1.0
        s1 = -1.0
        s2 = -1.0
        nb = 0.0
    eq["is_valid"] = m
    eq["is_valid2"] = s1
    eq["is_valid3"] = s2
    eq["is_valid4"] = nb
    return eq

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
        self.max_examples_to_tb = 20
        self.encoder = (
            self.modules["encoder"].module
            if self.params.multi_gpu
            else self.modules["encoder"]
        )
        self.decoder = (
            self.modules["decoder"].module
            if self.params.multi_gpu
            else self.modules["decoder"]
        )
        self.EncDecModel=FullEncDecModel(self.encoder, self.decoder, self.env, self.params)
        self.encoder.eval()
        self.decoder.eval()
        self.EncDecModel.eval()

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

        with torch.no_grad():
            for data_type in ["valid"]:
                for task in params.tasks:
                    self.enc_dec_step(data_type, task, scores)

        return scores
    def display_logs(self, logs, offset, f_export):  # FC A revoir
        """
        Display detailed results about success / fails.
        """
        if self.params.eval_verbose == 0:
            return

        for i, res in sorted(logs.items()):
            n_valid = sum([int(v) for _, _, _, _, v in res["hyps"]])
            n_valid_sign = sum([int(s) for _, _, _, s, _ in res["hyps"]])
            n_valid_mag = sum([int(m) for _, _, m, _, _ in res["hyps"]])
            s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\n"
            s += f"src={res['src']}\ntgt={res['tgt']}\n"
            for hyp, score, validmag, validsign, valid in res["hyps"]:
                # if score is None:
                s += f"{int(valid)} {hyp}\n"
                # else:
                #    s += f"{int(valid)} {score :.3e} {hyp}\n"

            if self.params.eval_verbose_print:
                logger.info(s)
            f_export.write(s + "\n")
            f_export.flush()
        return

    def do_interpretability_for_example(self, src_captum, len1_captum, tgt_captum, len2_captum, y_captum, pred_mask_captum,
                                        counter, tag):
        params = self.params
        env = self.env
        EncDecModel = self.EncDecModel
        encoder = self.EncDecModel.encoder
        trainer = self.trainer


        if (params.do_captum or params.return_scores) and params.eval_only:
            # put captum stuff here for now: attribute per example
            # attribute token in tgt
            # captum needs input batch size first

            if params.do_captum:
                lig_enc = LayerIntegratedGradients(EncDecModel.forward_captum, EncDecModel.encoder.embeddings)
                src_ref = encoder.get_ref_embedding(x=src_captum.transpose(0, 1), lengths=len1_captum, causal=False)
                # Attribute all but start token
                attrs_total = torch.zeros_like(src_captum.transpose(0, 1)).squeeze().double()
                delta_total = torch.zeros(1118).cuda()
                best_scores = []
                best_tokens = []
                for j in range(len2_captum[0] - 1):
                    # captum needs us to run again but only generate a single token, so just do another forward pass but only return the relevant token
                    word_scores_captum = EncDecModel.forward_captum(src_captum, len1_captum, tgt_captum,
                                                                    len2_captum,
                                                                    y_captum, pred_mask_captum, j)
                    attributions_enc_0, delta_0 = lig_enc.attribute(inputs=src_captum,
                                                                    baselines=src_ref.transpose(0, 1),
                                                                    additional_forward_args=(
                                                                        len1_captum, tgt_captum, len2_captum,
                                                                        y_captum,
                                                                        pred_mask_captum, j),
                                                                    internal_batch_size=1,
                                                                    return_convergence_delta=True)
                    attributions_enc_sum = summarize_attributions(attributions_enc_0.squeeze())
                    attrs_total += attributions_enc_sum
                    delta_total += delta_0
                    best_scores.append(torch.softmax(word_scores_captum[0], dim=0).cpu().tolist())
                    best_tokens.append(torch.argmax(word_scores_captum).cpu().tolist())
                    # args are: word_attributions,pred_prob, pred_class, true_class, attr_class, attr_score, raw_input_ids, convergence_score
                    thisviz = viz.VisualizationDataRecord(
                        attributions_enc_sum,
                        torch.max(torch.softmax(word_scores_captum[0], dim=0)),
                        env.id2word[torch.argmax(word_scores_captum).cpu().tolist()].replace("<", "&lt;").replace(
                            ">",
                            "&gt;"),
                        env.id2word[tgt_captum[0, j + 1].cpu().tolist()].replace("<", "&lt;").replace(">", "&gt;"),
                        env.id2word[torch.argmax(word_scores_captum).cpu().tolist()].replace("<", "&lt;").replace(
                            ">",
                            "&gt;"),
                        attributions_enc_sum.sum(),
                        [env.id2word[i] for i in src_captum.squeeze().tolist()],
                        delta_0)
                    myviz = viz.visualize_text([thisviz])
                    # write to the pattern buffer so we can add it as an image in tensorboard
                    img = imgkit.from_string(myviz.data, False)
                    buf = BytesIO(img)
                    pimg = F.pil_to_tensor(Image.open(buf))
                    trainer.writer.add_image(f"{tag}_{counter}/captum_attr", pimg, global_step=counter * 10 + j)

                # now do the total
                thisviz = viz.VisualizationDataRecord(
                    attrs_total,
                    sum(best_scores),
                    "&lt;s&gt;" + "".join([env.id2word[tok] for tok in best_tokens]).replace("<", "&lt;").replace(
                        ">",
                        "&gt;"),
                    "".join([env.id2word[tok] for tok in tgt_captum.squeeze().tolist()]).replace("<",
                                                                                                 "&lt;").replace(
                        ">", "&gt;"),
                    "&lt;s&gt;" + "".join([env.id2word[tok] for tok in best_tokens]).replace("<", "&lt;").replace(
                        ">",
                        "&gt;"),
                    attrs_total.sum(),
                    [env.id2word[i] for i in src_captum.squeeze().tolist()],
                    delta_total)
                myviz = viz.visualize_text([thisviz])
                # write to the pattern buffer so we can add it as an image in tensorboard
                img = imgkit.from_string(myviz.data, False)
                buf = BytesIO(img)
                pimg = F.pil_to_tensor(Image.open(buf))
                trainer.writer.add_image(f"{tag}_{counter}/captum_attr", pimg, global_step=counter * 10 + 9)

            if params.return_scores:
                # get the attn maps
                my_cmap = cm.coolwarm
                word_scores_captum, enc_attn_scores, dec_attn_scores = EncDecModel.forward_captum(src_captum,
                                                                                                  len1_captum,
                                                                                                  tgt_captum,
                                                                                                  len2_captum,
                                                                                                  y_captum,
                                                                                                  pred_mask_captum,
                                                                                                  -999)
                for l, layerscore in enumerate(enc_attn_scores):
                    layerscore = layerscore.cpu().squeeze()
                    my_min = 100000000
                    my_max = -100000000
                    for headscore in layerscore:
                        if torch.max(headscore) > my_max: my_max = torch.max(headscore)
                        if torch.min(headscore) < my_min: my_min = torch.min(headscore)
                    for h, headscore in enumerate(layerscore):
                        plt.figure()
                        fig, ax = plt.subplots(figsize=(12, 12))
                        my_show = ax.matshow(headscore, cmap=my_cmap, vmin=my_min, vmax=my_max)
                        ax.set_xticks(np.arange(len(src_captum.squeeze().tolist())),
                                      labels=[env.id2word[i] for i in src_captum.squeeze().tolist()])
                        ax.set_yticks(np.arange(len(src_captum.squeeze().tolist())),
                                      labels=[env.id2word[i] for i in src_captum.squeeze().tolist()])
                        # We change the fontsize of minor ticks label
                        ax.tick_params(axis='both', which='major', labelsize=5)
                        ax.tick_params(axis='both', which='minor', labelsize=5)
                        fig.colorbar(my_show, ax=ax)
                        plt.tight_layout()
                        buf = BytesIO()
                        plt.savefig(buf, format='png')
                        pimg = F.pil_to_tensor(Image.open(buf))
                        trainer.writer.add_image(f"{tag}_{counter}/enc_attn_layer_{l}", pimg,
                                                      global_step=l * 10 + h)
                        plt.close()

                for l, layerscore in enumerate(dec_attn_scores):
                    layerscore = layerscore.cpu().squeeze()
                    fig = plt.figure(figsize=(12, 12))
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
                        ax.set_xticks(np.arange(len(src_captum.squeeze().tolist())),
                                      labels=[env.id2word[i] for i in src_captum.squeeze().tolist()])
                        ax.set_yticks(np.arange(len(tgt_captum.squeeze().tolist())),
                                      labels=[env.id2word[i] for i in tgt_captum.squeeze().tolist()])
                        # We change the fontsize of minor ticks label
                        ax.tick_params(axis='both', which='major', labelsize=5)
                        ax.tick_params(axis='both', which='minor', labelsize=5)
                    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
                    fig.colorbar(my_show, cax=cb_ax)
                    plt.tight_layout()
                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    pimg = F.pil_to_tensor(Image.open(buf))
                    trainer.writer.add_image(f"{tag}_{counter}/dec_attn_layer_{l}", pimg,
                                                  global_step=l)
                    plt.close()

            counter += 1
        return counter

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

        # evaluation details
        if params.eval_verbose:
            if params.beam_eval:
                eval_path = os.path.join(
                    params.dump_path, f"eval.beam.{data_type}.{task}.{scores['epoch']}"
                )
            else:
                eval_path = os.path.join(
                    params.dump_path, f"eval.{data_type}.{task}.{scores['epoch']}"
                )
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        # stats
        xe_loss = 0
        n_valid = torch.zeros(10000, dtype=torch.long)
        n_total = torch.zeros(10000, dtype=torch.long)
        n_valid_additional = np.zeros(1 + len(env.additional_tolerance), dtype=int)
        n_perfect_match = 0
        n_correct = 0
        n_valid_d1 = 0
        n_valid_d2 = 0
        n_valid_nb = 0

        #for interp
        do_interp = True
        zero_coef_count = 0
        all_zero_inputs_count = 0
        scmi_count = 0
        simc_count = 0
        both_incorrect = 0
        both_correct = 0

        # iterator
        iterator = self.env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=params.eval_size,
        )
        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2), nb_ops in iterator:
            y, pred_mask = get_y_and_mask(x2, len2)
            # cuda
            x1_, len1_, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

            bs = len(len1)
            word_scores,loss = self.EncDecModel.forward(x1_, len1_, x2, len2,y,pred_mask)

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()
            n_perfect_match += valid.sum().item()
            # export evaluation details
            beam_log = {}
            # stats
            xe_loss += loss.item() * len(y)
            n_valid.index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

            for i in range(len(len1)):
                src = env.idx_to_infix(x1[1 : len1[i] - 1, i].tolist(), True)
                tgt = env.idx_to_infix(x2[1 : len2[i] - 1, i].tolist(), False)
                #log it if it's good
                if valid[i]:
                    beam_log[i] = {"src": src, "tgt": tgt, "hyps": [(tgt, None, True, True, True)]}

                if do_interp:
                    #currently, this only works for sign-first outputs.
                    len1_captum = torch.IntTensor([len1[i]])
                    len2_captum = torch.IntTensor([len2[i]])
                    len1_captum, len2_captum = to_cuda(len1_captum, len2_captum)
                    # already on cuda
                    src_captum = torch.unsqueeze(x1[:len1[i], i], dim=1)
                    tgt_captum = torch.unsqueeze(x2[:len2[i], i], dim=1)
                    y_captum, pred_mask_captum = get_y_and_mask(tgt_captum, len2_captum)
                    src_captum = src_captum.transpose(0, 1)
                    tgt_captum = tgt_captum.transpose(0, 1)
                    pred_mask_captum = pred_mask_captum.transpose(0, 1)
                    src_captum, tgt_captum = to_cuda(src_captum, tgt_captum)
                    this_valid = False

                    #if t is false for example i, we got the sign wrong
                    this_valid_sign = t[0,i]
                    this_valid_mag = torch.all(t[1:len2[i]-1, i])
                    if this_valid_mag and this_valid_sign: this_valid = True

                    #do viz for greedy generation results only
                    # do viz for greedy generation results only
                    if is_all_zeroes(re.sub(r'[^\w]', '', tgt)) and (zero_coef_count < self.max_examples_to_tb):
                        zero_coef_count = self.do_interpretability_for_example(src_captum,len1_captum, tgt_captum,
                                                                               len2_captum, y_captum, pred_mask_captum,
                                                                               zero_coef_count, "coef_zero")
                    if is_all_zeroes(re.sub(r'[^\w]', '', src)) and (all_zero_inputs_count < self.max_examples_to_tb):
                        all_zero_inputs_count = self.do_interpretability_for_example(src_captum,len1_captum, tgt_captum,
                                                                                     len2_captum, y_captum,
                                                                                     pred_mask_captum,
                                                                                     all_zero_inputs_count,
                                                                                     "input_zero")
                    if (not this_valid_mag) and (this_valid_sign) and (scmi_count < self.max_examples_to_tb):
                        scmi_count = self.do_interpretability_for_example(src_captum,len1_captum, tgt_captum,
                                                                          len2_captum, y_captum, pred_mask_captum,
                                                                          scmi_count, "sign_corr_only")
                    if (this_valid_mag) and (not this_valid_sign) and (simc_count < self.max_examples_to_tb):
                        simc_count = self.do_interpretability_for_example(src_captum,len1_captum, tgt_captum,
                                                                          len2_captum, y_captum, pred_mask_captum,
                                                                          simc_count,"mag_corr_only")
                    if (not this_valid_mag) and (not this_valid_sign) and (both_incorrect < self.max_examples_to_tb):
                        both_incorrect = self.do_interpretability_for_example(src_captum,len1_captum, tgt_captum,
                                                                              len2_captum, y_captum, pred_mask_captum,
                                                                              both_incorrect,"incorrect")
                    if this_valid and (both_correct < self.max_examples_to_tb):
                        both_correct = self.do_interpretability_for_example(src_captum,len1_captum, tgt_captum,
                                                                            len2_captum, y_captum, pred_mask_captum,
                                                                            both_correct, "correct")

                    if (both_correct > self.max_examples_to_tb) and (zero_coef_count > self.max_examples_to_tb) \
                            and (all_zero_inputs_count > self.max_examples_to_tb) and (
                            scmi_count > self.max_examples_to_tb) \
                            and (simc_count > self.max_examples_to_tb):
                        do_interp = False


            # invalid top-1 predictions - check if there is a solution in the beam
            invalid_idx = (1 - valid).nonzero().view(-1)
            logger.info(
                f"({n_total.sum().item()}/{eval_size}) Found "
                f"{bs - len(invalid_idx)}/{bs} valid top-1 predictions. "
                f"Generating solutions ..."
            )

            # continue if the whole batch is correct. if eval_verbose, perform
            # a full beam search, even on correct greedy generations
            if valid.sum() == len(valid) and params.eval_verbose == 1:
                self.display_logs(beam_log, offset=n_total.sum().item() - bs,f_export=f_export)
                continue

            # generate
            encoded = self.encoder("fwd", x=x1_, lengths=len1_, causal=False)

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
                generated, _ = self.decoder.generate(
                    encoded.transpose(0, 1),
                    len1_,
                    max_len=max_beam_length,
                )
                generated=generated.transpose(0, 1)

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
                    end_idx = generated[i][1:].tolist().index(env.eos_index)+1
                    inputs.append(
                        {
                            "i": i,
                            "src": x1[1 : len1[i] - 1, i].tolist(),
                            "tgt": x2[1 : len2[i] - 1, i].tolist(),
                            "hyp": generated[i][1:end_idx].tolist(),
                            "task": task,
                        }
                    )

            # check hypotheses with multiprocessing
            outputs = []
            if params.windows is True:
                for inp in inputs:
                    outputs.append(check_hypothesis(inp))
            else:
                with ProcessPoolExecutor(max_workers=20) as executor:
                    for output in executor.map(check_hypothesis, inputs, chunksize=1):
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

                curr_correct = 0
                curr_d1 = 0
                curr_d2 = 0
                curr_nb = 0
                curr_valid = 0
                curr_additional = np.zeros(1 + len(env.additional_tolerance), dtype=int)

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
                    is_valid2 = gen["is_valid2"]
                    is_valid3 = gen["is_valid3"]
                    is_valid4 = gen["is_valid4"]
                    is_b_valid = is_valid >= 0.0 and is_valid < env.float_tolerance
                    is_valid_mag = is_valid2 >= 0.0 and is_valid2 < env.float_tolerance
                    is_valid_sign = is_valid3 >= 0.0 and is_valid3 < env.float_tolerance
                    if not valid[i]:
                        if is_valid2 >= 0:
                            curr_d1 = 1
                        if is_valid3 >= 0:
                            curr_d2 = 1

                        if is_valid >= 0.0:
                            curr_correct = 1
                            if is_valid4 > curr_nb:
                                curr_nb = is_valid4
                            for k, tol in enumerate(env.additional_tolerance):
                                if is_valid < tol:
                                    curr_additional[k] = 1
                            if is_valid < env.float_tolerance:
                                curr_valid = 1

                    # update beam log
                    beam_log[i]["hyps"].append((gen["hyp"], None, is_valid_mag, is_valid_sign, is_b_valid))
                    # gen["score"], is_b_valid))
                if not valid[i]:
                    n_correct += curr_correct
                    n_valid_d1 += curr_d1
                    n_valid_d2 += curr_d2
                    n_valid_nb += curr_nb
                    for k, tol in enumerate(env.additional_tolerance):
                        n_valid_additional[k] += curr_additional[k]
                    valid[i] = curr_valid
                    n_valid[nb_ops[i]] += curr_valid

            # valid solutions found with beam search
            logger.info(
                f"    Found {valid.sum().item()}/{bs} solutions in beam hypotheses. "
            )

            # export evaluation details
            if params.eval_verbose:
                assert len(beam_log) == bs
                self.display_logs(beam_log, offset=n_total.sum().item() - bs, f_export=f_export)

        logger.info(
            f"Found {n_valid_d1} correct abs, and {n_valid_d2} correct signs"
        )
    
        # evaluation details
        if params.eval_verbose:
            f_export.close()
            logger.info(f"Evaluation results written in {eval_path}")
    
        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(
            f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) "
            f"equations were evaluated correctly."
        )
    
        # compute perplexity and prediction accuracy
        assert _n_total == eval_size
        scores[f"{data_type}_{task}_xe_loss"] = xe_loss / _n_total
        scores[f"{data_type}_{task}_acc"] = 100.0 * _n_valid / _n_total
        scores[f"{data_type}_{task}_perfect"] = 100.0 * n_perfect_match / _n_total
        scores[f"{data_type}_{task}_correct"] = (
            100.0 * (n_perfect_match + n_correct) / _n_total
        )
        scores[f"{data_type}_{task}_acc_d1"] = (
            100.0 * (n_perfect_match + n_valid_d1) / _n_total
        )
        scores[f"{data_type}_{task}_acc_d2"] = (
            100.0 * (n_perfect_match + n_valid_d2) / _n_total
        )
        scores[f"{data_type}_{task}_acc_nb"] = (
            100.0 * (n_perfect_match + n_valid_nb) / _n_total
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
            scores[f"{data_type}_{task}_acc_{e}"] = (
                100.0 * n_valid[i].item() / max(n_total[i].item(), 1)
            )
            if n_valid[i].item() > 0:
                logger.info(
                    f"{e}: {n_valid[i].item()} / {n_total[i].item()} "
                    f"({100. * n_valid[i].item() / max(n_total[i].item(), 1)}%)"
                )

    def format_and_draw_inferences(self, src_string, tgt_string, hyp_list, tag, step):
        """
        Takes a matplotlib figure handle and converts it using
        canvas and string-casts to a numpy array that can be
        visualized in TensorBoard using the add_image function
        """
        data = {'Inputs': [src_string.replace("+"," +").replace("-"," -")],
                'True_Coef': [tgt_string],
                 'Predicted_Coef':[hyp_list[0][0]]}

        table = f"""
            | Inputs = {data["Inputs"][0]} | Pred_Coef = {data["Predicted_Coef"][0]} | True_Coef = {data["True_Coef"][0]} |
        """
        # Add table in numpy "text" to TensorBoard writer
        self.trainer.writer.add_text(tag, table, step)
        plt.close()


#wrapper to get the full model as one module, for captum. may be a cleaner way to do this
class FullEncDecModel(torch.nn.Module):
    def __init__(self, enc, dec, env, params):
        super().__init__()
        self.encoder = enc
        self.decoder = dec
        self.env = env
        self.params = params
    def forward(self, x1_, len1_, x2, len2, y, pred_mask):
        encoded = self.encoder("fwd", x=x1_, lengths=len1_, causal=False)
        if self.params.eval_only:
            inputs_embs_outputs_to_tsv(self.env, x1_,x2,encoded,f"./runs_eval/{self.params.exp_id}_encodings.tsv",
                                       f"./runs_eval/{self.params.exp_id}_metadata.tsv")
        decoded = self.decoder(
            "fwd",
            x=x2,
            lengths=len2,
            causal=True,
            src_enc= encoded.transpose(0, 1),
            src_len=len1_,
        )
        word_scores, loss = self.decoder(
            "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
        )
        return word_scores, loss
    def forward_captum(self, x1_, len1_, x2, len2, y, pred_mask,token_num):
        # get out of batch first form
        x1_ = x1_.transpose(0, 1)
        x2 = x2.transpose(0, 1)
        pred_mask = pred_mask.transpose(0, 1)
        encoded = self.encoder("fwd", x=x1_, lengths=len1_, causal=False)
        enc_attn_scores = self.encoder.all_scores
        decoded = self.decoder(
            "fwd",
            x=x2,
            lengths=len2,
            causal=True,
            src_enc=encoded.transpose(0, 1),
            src_len=len1_,
        )
        dec_attn_scores = self.decoder.all_scores
        word_scores, loss = self.decoder(
            "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
        )
        if token_num > -1:
            return word_scores[token_num,:]
        else:
            return word_scores, enc_attn_scores, dec_attn_scores