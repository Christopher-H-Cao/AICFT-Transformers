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
python train.py --reload_data cfts,./sample_data/su2_cfts.data.train,./sample_data/su2_cfts.data.valid,./sample_data/su2_cfts.data.valid --max_epoch 20 --n_enc_layers 1 --n_dec_layers 1 --num_workers 1 --eval_verbose 1 --batch_size 512 --batch_size_eval 10 --enc_emb_dim 512 --dec_emb_dim 512 --amp 1 --fp16 True

The allowed values of the "operation" flag are ["cc","cc_decimal","ADE", "ADE_decimal"]. The 'decimal' indicator tells us whether the input data is given as floats (and we need to convert it to rationals),
or if it is rational already. The default value of this flag is "cc".

Eval-only Example:
python train.py --eval_only --eval_data ./sample_data/su2_cfts.data.valid --eval_from_exp ./checkpoint/garrett/dumped/debug/m0cwik7xz5 --eval_verbose 1 

# Docker
All package dependencies are listed in requirements.txt. The relatively-small dockerfile provided can be used to run the repo on a setup such as UW's CHTC (you will need to copy the code and data in for now).In general, though, this is not needed for local runs!

# Tensorboard
Metrics are logged on wandb- create a weights and biases account to follow trainings as they happen. If no wandb account is found, logging will happen on tensorboard instead.


## Running on CHTC

### 1. Environment setup

Activate conda on the CHTC node:
```bash
eval "$(/home/hcao39/anaconda3/bin/conda shell.bash hook)"
conda activate py3
```

### 2. Stage data (large files)

Create a tarball of the repository (excluding git history) and copy it to CHTC staging:
```bash
tar --exclude-vcs -czvf runjob_package_cft.tar Chromoboot_CFT
scp runjob_package_cft.tar hcao39@transfer.chtc.wisc.edu:/staging/hcao39/
```

To browse staging:
```bash
ssh hcao39@transfer.chtc.wisc.edu
cd /staging/hcao39
ls
```

### 3. Prepare and submit job

Upload the `.sh` and `.sub` scripts to the submit node:
```bash
scp run_cft.sh run_cft.sub hcao39@ap2002.chtc.wisc.edu:/home/hcao39/
ssh hcao39@ap2002.chtc.wisc.edu
condor_submit run_cft.sub
```

Monitor and manage jobs:
```bash
condor_q           # check job status
condor_rm JOBID    # remove a job
```

### 4. Training commands

**Standard KM dataset (c < 50):**
```bash
python train.py \
  --reload_data cfts,./sample_data/cft_KM_3M_l2-5_cleq50_out.csv.train,./sample_data/cft_KM_3M_l2-5_cleq50_out.csv.valid,./sample_data/cft_KM_3M_l2-5_cleq50_out.csv.valid \
  --max_epoch 300 --n_enc_layers 2 --n_dec_layers 2 \
  --n_enc_heads 8 --n_dec_heads 8 \
  --num_workers 1 --eval_verbose 1 \
  --batch_size 512 --batch_size_eval 10 \
  --enc_emb_dim 256 --dec_emb_dim 256 \
  --amp 1 --fp16 True
```

**With priming data (e.g. adding coset or higher-c examples):**
```bash
python3 combine.py cft_KM+coset_l2-5_priming30.csv cft_KM_3M_l2-5_cleq50_out.csv.train cft_KM+coset_l2-5_priming30_train.csv
# then pass the combined file to --reload_data
```

### 5. Checking results

```bash
# Check accuracy in a single eval file
grep "Equation" eval.valid.cfts.49 | grep "\(1/1\)" | wc -l

# Find the best epoch across all eval files
for n in {1..100}; do
  count=$(grep "Equation" "eval.valid.cfts.$n" 2>/dev/null | grep "\(1/1\)" | wc -l)
  echo "$n $count"
done | sort -k2 -rn | head -5

# Check checkpoint directory
ls -lrth ./checkpoint/hcao39/dumped/debug
```

### 6. Data pipeline reference

Move processed data into the repo:
```bash
mv your_dataset.csv ~/Chromoboot_CFT/sample_data/
```

Full preprocessing sequence:
```bash
python make_cft_data_from_csv.py \
  --path_to_csv sample_data/your_dataset \
  --path_to_outfile sample_data/your_dataset_out \
  --num_rows 1000 --num_cols 10 --target_variable 'cc'
shuf sample_data/your_dataset_out.csv --output sample_data/your_dataset_out.csv
python3 split_data.py --data_path sample_data/your_dataset_out --valid_set_size 10000 --no_test True
```
