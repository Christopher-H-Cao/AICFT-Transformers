# ChromoBoot

## Setup
From a fresh conda environment, do 
pip install -r requirements.txt

to install any required packages that you may not have. Make sure you are using python 3.10 (my version is 3.10.10) as pytorch does not yet support python 3.11!

## Dataset building

*make_cft_data_from_csv.py:* 

`python make_cft_data_from_csv.py --path_to_csv {YOUR CSV} --path_to_output_file {OUTPUT FILE} --num_rows {rows} --num_cols {columns} --target_variable {'cc' or 'ade'}`

This code processes a csv of CFTs, with the CC first. You can truncate it to only n terms of each cft with 'num_cols', or take only the first n cfts with 'n_rows'.
This script is totally agnostic to the format of input data (decimal or rational).
You can also modify this script to calculate A,D,E or CC using k (should just be row number in CSV).

*shuffling and concatenating:*

`shuf {OUTPUT FILE} > {OUTPUT FILE}`

Shuffle the file. `make_cft_data_from_csv.py` outputs this line for you to copy). Note that for macs with GNU installed but not by default, need to use `gshuf` instead of `shuf`. 

*split_data.py:*

 `python split_data.py  --data_path output_file.prefix --valid_set_size 1000 --no_test True`

Build train, validation and test set. First parameter is the name of the file to split (produced at the previous step), second is the number of elements in the train and test set .
We end up with three files for each dataset, that will be used by parameter `reload_data`of the transformer code.

## Running the model

Training Example:
python train.py --reload_data cfts,./sample_data/su2_cfts.data.train,./sample_data/su2_cfts.data.valid,./sample_data/su2_cfts.data.valid --max_epoch 20 --n_enc_layers 1 --n_dec_layers 1 --num_workers 1 --eval_verbose 1 --batch_size 512 --batch_size_eval 10 --enc_emb_dim 512 --dec_emb_dim 512 --amp 1 --fp16 True

The allowed values of the "operation" flag are ["cc","cc_decimal","ADE", "ADE_decimal"]. The 'decimal' indicator tells us whether the input data is given as floats (and we need to convert it to rationals),
or if it is rational already. The default value of this flag is "cc".

Eval-only Example:
python train.py --eval_only --eval_data ./sample_data/su2_cfts.data.valid --eval_from_exp ./checkpoint/garrett/dumped/debug/m0cwik7xz5 --eval_verbose 1 

# Docker
All package dependencies are listed in requirements.txt. The relatively-small dockerfile provided can be used to run the repo on a setup such as UW's CHTC (you will need to copy the code and data in for now).In general, though, this is not needed for local runs!

# Tensorboard
Metrics are logged on wandb- create a weights and biases account to follow trainings as they happen. If no wandb account is found, logging will happen on tensorboard instead.
