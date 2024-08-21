# ChromoBoot

## Setup
From a fresh conda environment, do 
conda env create --file conda_reqs.txt

to install any required packages that you may not have. Make sure you are using python 3.10 (my version is 3.10.10) as pytorch does not yet support python 3.11!

## Dataset building

*make_cft_data_from_csv.py:* 

`python make_cft_data_from_csv.py --path_to_csv {YOUR CSV} --path_to_outfile {OUTPUT FILE} --num_rows {rows} --num_cols {columns} --target_variable {'cc' or 'ade'}`

This code processes a csv of CFTs, with the CC first. You can truncate it to only n terms of each cft with 'num_cols', or take only the first n cfts with 'n_rows'.
This script is totally agnostic to the format of input data (decimal or rational).
You can also modify this script to calculate A,D,E or CC using k (should just be row number in CSV).

*shuffling and concatenating:*

<<<<<<< HEAD
`shuf {OUTPUT FILE} > {OUTPUT FILE}`
=======
`shuf {OUTPUT FILE} --output {OUTPUT FILE}`
>>>>>>> ad6217c (clean up)

Shuffle the file. `make_cft_data_from_csv.py` outputs this line for you to copy). Note that for macs with GNU installed but not by default, need to use `gshuf` instead of `shuf`. 

*split_data.py:*

 `python split_data.py  --data_path output_file.prefix --valid_set_size 1000 --no_test True`

Build train, validation and test set. First parameter is the name of the file to split (produced at the previous step), second is the number of elements in the train and test set .
We end up with three files for each dataset, that will be used by parameter `reload_data`of the transformer code.
"--no_test" only generates two files, in case we do not want to do hyperparameter tuning (this is a decent option for experiments).

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


#to run on CHTC:
To run this on CHTC on a new dataset:

First, process the data into the desired input-output format using "make_cft_data_from_csv.py".
Shuffle the output file and split it into a file.train and a file.valid (and possibly a file.test as well).
For early experiments, I recommend a 90%-10% train/valid split with no test.
Put these files into a directory. I recommend keeping this outside this repo, in case files get large:
accidentally checking large files into git can cause major problems!


To run, make a tarball containing the repository folder and your data folder:
`tar --exclude-vcs -czvf runcft_package.tar ChromoBoot_CFT {YOUR_DATA_FOLDER}`
IMPORTANT: If you've run the model locally, make sure to delete the "wandb" and "checkpoint" directories before doing this!

Copy the tarball to the transfer node on CHTC. I use 'staging' for this and recommend you do as well.

Edit the .sub and .sh scripts as needed. Add your WANDB_API key.
Make sure that the .sh script unpacks the tarball, cds into the ChromoBoot_CFT directory, copies the data into it if it is not there already,
and that the train.py command points to the place the data is located in the ChromoBoot_CFT directory.

Copy the .sub and .sh scripts to the submit node on CHTC.

Edit them as needed- make sure the .sub script correctly points to the .sh one.
do: `condor_submit run_training_cfts.sub`

Wait
