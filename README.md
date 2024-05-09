# ChromoBoot

## Setup
From a fresh conda environment, do 
pip install -r requirements.txt

to install any required packages that you may not have. Make sure you are using python 3.10 (my version is 3.10.10) as pytorch does not yet support python 3.11!

## Dataset building
*generate_data.py:* 

`python generate_data.py input_path input_file output_file nr_loops base {mode}`
optional: 
argv[5]: base to put coef in. 0 == "raw"
argv[6]: 'b', 'zero', 'quad': binary output (zero/non zero), create zero coeff file, quad format

last argv[]: 's', 'is' : export sign only, export abs + sign (sign at the end), only get one quad/triple/quad+triple

python generate_data.py --input_file /private/home/fcharton/Amplitudes/ML/EZ_symb_new_norm --output_file /checkpoint/fcharton/dumped/chromoboost/loop4 --loops 4 --case zero --generate_relations True --rels_output_path ./relations (Note: do not add 'relations'; just the folder name; otherwise the output relation files will be named 'relationsrel_instances_XX.json'.)

This will generate two files: output_file.data and output_file.zero, with the same number of lines. Each line as a number, the 2L abcdef tokens separated by blanks, and the corresponding coeff written as `sign base1000_repr` (e.g. 2022 is `+ 2 22`)

*shuffling and concatenating:*

`cat output_file.data output_file.zero | shuf > output_file.prefix`

Concat the files and shuffle them. `generate_data.py` outputs this line for you to copy). Note that for macs with GNU installed but not by default, need to use `gshuf` instead of `shuf`. 

Over the EZ_symb_new_norm files, we get the following dataset sizes (`wc -l *.prefix`) 

with zeroes
         24 loop2.prefix
       1272 loop3.prefix
      22416 loop4.prefix
     527760 loop5.prefix
    9832932 loop6.prefix

without zeroes
         12 loop2.prefix
        636 loop3.prefix
      11208 loop4.prefix
     263880 loop5.prefix
    4916466 loop6.prefix

*split_data.py:*

 `python split_data.py  --data_path /checkpoint/fcharton/dumped/chromoboost/loop4.prefix --valid_set_size 10000 --no_test True`

Build train, validation and test set. First parameter is the name of the file to split (produced at the previous step), second is the number of elements in the train and test set (I used 1000 for loop4, and 10000 for the others). We end up with three files for each dataset, that will be used by parameter `reload_data`of the transformer code.
no_test exports only valid set (no test set, not used at present)

      1000 loop4.prefix.test
     20416 loop4.prefix.train
      1000 loop4.prefix.valid

     10000 loop5.prefix.test
    507760 loop5.prefix.train
     10000 loop5.prefix.valid

      10000 loop6.prefix.test
    9812932 loop6.prefix.train
      10000 loop6.prefix.valid

To generate the date for the "parents" study, call the generate_recur_data script in a similar manner.
## Running the model

Training Example:
python train.py --reload_data boots,../ChromoBoot_data/processed_data/loop4.prefix.train,../ChromoBoot_data/processed_data/loop4.prefix.valid,../ChromoBoot_data/processed_data/loop4.prefix.valid
--max_epoch 50 --n_enc_layers 2 --n_dec_layers 2 --num_workers 1 --eval_relations True --relations_path ../rels/

Eval Example:
python train.py --eval_only --eval_data ../ChromoBoot_data/processed_data/loop6_recur.prefix.valid --eval_from_exp ./checkpoint/garrett/dumped/debug/m0cwik7xz5 --eval_verbose 1 

# Docker
All package dependencies are listed in requirements.txt. The relatively-small dockerfile provided can be used to run the repo on a setup such as UW's CHTC (you will need to copy the code and data in for now).In general, though, this is not needed for local runs!

# Tensorboard
Metrics are logged in tensorboard format in the "runs" directory, or in the "xp" directory if you would like to use the ReadXP notebook. To use the tensorboard viewer, simply call "tensorboard --logdir runs_train".
NOTE: to view the input parameters table in the HPARAMS tab, you also need to specify at least one metric in the validation_metrics args! I recommend just using the full default set.
20 examples of perfect match, sign match but coeff different, etc. samples are logged on the tensorboard. You can see these at train or eval time by calling with the "eval_verbose" flag set to 1. 
