#!/bin/bash
echo "Hello CHTC from Job running on `hostname`"
echo "training- 6loop coef from word!"

# This is secret and shouldn't be checked into version control
export WANDB_API_KEY="334225e5e6323c8ba4460abb984e0ae6c7590a12"
# Name and notes optional
export WANDB_NAME=""
export WANDB_NOTES="6loop, with rels, baseline"

# Only needed if you don't check in the wandb/settings file
export WANDB_ENTITY="ai_amplitudes"
export WANDB_PROJECT="ChromoBoot"

cp -r /staging/gmerz2/runjob_package_42723.tar .
tar -xzvf runjob_package_42723.tar
# Check if checkpoint tarball exists; if so then unpack
checkpoint_tarball=checkpoint_${4}.tar.gz
if [[ -f "$checkpoint_tarball" ]]; then
tar -xzf $checkpoint_tarball
rm -rf ChromoBoot/checkpoint
rm -rf ChromoBoot/runs_train
mv ./checkpoint ChromoBoot
mv ./runs_train ChromoBoot
mv ./wandb ChromoBoot
fi
cd ChromoBoot
mkdir data
mkdir relations
cp -r ../ChromoBoot_data/processed_data/coef_from_word/6loop/train_valid/*loop6* ./data
cp -r ../ChromoBoot_data/processed_data/coef_from_word/relations_6loop/* ./relations
timeout 4h python train.py --reload_data boots,./data/loop6.prefix.train,./data/loop6.prefix.valid,./data/loop6.prefix.valid --max_epoch 200 --n_enc_layers $2 --n_dec_layers $2 --num_workers 1 --eval_verbose 1 --eval_relations True --relations_path ./relations --resume_wandb True --batch_size 512 --batch_size_eval 512 --env_base_seed	$3 --exp_id $4 --max_output_len 5 --hardcode_trivial_zeroes True --enc_emb_dim 512 --dec_emb_dim 512 --amp 1 --fp16 True

timeout_exit_status=$?
# Uses the bash notation of `$?` to call the exit value of the last executed command
# and to save it in a variable called `timeout_exit_status`.


if [ $timeout_exit_status -eq 124 ]; then
    tar -czf $checkpoint_tarball ./checkpoint ./runs_train ./wandb
    mv $checkpoint_tarball ..
    exit 85
fi

mv ./checkpoint ./checkpoint_6loop
mv ./runs_train ./runs_train_6loop
mv ./wandb ./wandb_6loop

tar -czvf training_output_6loop_coef_from_word_${2}layers_seed${3}_rels.tar ./checkpoint_6loop ./runs_train_6loop
cp -r training_output_6loop_*.tar /staging/gmerz2

rm -rf *

exit $timeout_exit_status
