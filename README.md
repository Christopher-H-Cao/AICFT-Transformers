# ChromoBoot

## Setup
From a fresh conda environment, do 
conda env create --file conda_reqs.txt

to install any required packages that you may not have. Make sure you are using python 3.10 (my version is 3.10.10) as pytorch does not yet support python 3.11!

## Dataset preprocessing

*shuffling and concatenating:*

`shuf {OUTPUT FILE} --output {OUTPUT FILE}`

*split_data.py:*

 `python split_data.py  --data_path output_file.prefix --valid_set_size 1000 --no_test True`

Build train, validation and test set. First parameter is the name of the file to split (produced at the previous step), second is the number of elements in the train and test set .
We end up with three files for each dataset, that will be used by parameter `reload_data`of the transformer code.
"--no_test" only generates two files, in case we do not want to do hyperparameter tuning (this is a decent option for experiments).

## Running the model

Training Example:
python train.py --reload_data cfts,./sample_data/cft_KM_l2-5_c50.csv.train,./sample_data/cft_KM_l2-5_c50.csv.valid,./sample_data/cft_KM_l2-5_c50.csv.valid --max_epoch 100 --n_enc_layers 2 --n_dec_layers 2 --num_workers 1 --eval_verbose 1 --batch_size 128 --batch_size_eval 10 --enc_emb_dim 256 --dec_emb_dim 256 --amp 1 --fp16 True 

# Docker
All package dependencies are listed in requirements.txt. The relatively-small dockerfile provided can be used to run the repo on a setup such as UW's CHTC (you will need to copy the code and data in for now).In general, though, this is not needed for local runs!

# Tensorboard
Metrics are logged on wandb- create a weights and biases account to follow trainings as they happen. If no wandb account is found, logging will happen on tensorboard instead.


## Running on Cluster

### 1. Environment setup

Activate conda on the CHTC node:
```bash
conda activate envname
```

### 2. Stage data (large files)

Create a tarball of the repository (excluding git history) and copy it to CHTC staging:
```bash
tar --exclude-vcs -czvf runjob_package_cft.tar Chromoboot_CFT
scp runjob_package_cft.tar clusterusername/folder
```

### 3. Training commands
You can add these lines to bash scripts as training examples. 

**Standard KM dataset (c < 50):**
```bash
python train.py --reload_data cfts,./sample_data/cft_KM_l2-5_c50.csv.train,./sample_data/cft_KM_l2-5_c50.csv.valid,./sample_data/cft_KM_l2-5_c50.csv.valid --max_epoch 100 --n_enc_layers 2 --n_dec_layers 2 --num_workers 1 --eval_verbose 1 --batch_size 128 --batch_size_eval 10 --enc_emb_dim 256 --dec_emb_dim 256 --amp 1 --fp16 True 
```

**With priming data (e.g. adding coset or higher-c examples):**
```bash
python3 combine.py cft_KM+coset_l2-5_primingxxx.csv cft_KM_l2-5_c50.csv.train cft_KM+coset_l2-5_primingxxx_train.csv
# then pass the combined file to --reload_data
```
