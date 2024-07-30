#!/bin/bash
echo "Hello CHTC from Job running on `hostname`"
echo "training- cft cc prediction"

# This is secret and shouldn't be checked into version control
export WANDB_API_KEY="334225e5e6323c8ba4460abb984e0ae6c7590a12"
# Name and notes optional
export WANDB_NAME=""
export WANDB_NOTES="ChromoBoot_CFT"

# Only needed if you don't check in the wandb/settings file
export WANDB_ENTITY="ai_amplitudes"
export WANDB_PROJECT="ChromoBoot_CFT"

#your tarball. Should contain ChromoBoot_CFT code and sample_data.
#Feel free to restructure as needed, but by the time the train command is passed,
#everything should be in a subdirectory of ChromoBoot_CFT.
cp -r your_tarball.tar .
tar -xzvf your_tarball.tar
# Check if checkpoint tarball exists; if so then unpack
checkpoint_tarball=checkpoint_${4}.tar.gz
if [[ -f "$checkpoint_tarball" ]]; then
tar -xzf $checkpoint_tarball
rm -rf ChromoBoot_CFT/checkpoint
rm -rf ChromoBoot_CFT/runs_train
mv ./checkpoint ChromoBoot_CFT
mv ./runs_train ChromoBoot_CFT
mv ./wandb ChromoBoot_CFT
fi
cd ChromoBoot_CFT
mkdir cft_data
cp -r ../cft_data/su2_cfts.* ./cft_data
timeout 4h python train.py --reload_data cfts,./cft_data/su2_cfts.data.train,./cft_data/su2_cfts.data.valid,./cft_data/su2_cfts.data.valid --max_epoch 20 --n_enc_layers $2 --n_dec_layers $2 --num_workers 1 --eval_verbose 1 --batch_size 512 --batch_size_eval 10 --enc_emb_dim 512 --dec_emb_dim 512 --amp 1 --fp16 True --epoch_size 900  --resume_wandb True --env_base_seed $3 --exp_id $4 
timeout_exit_status=$?
# Uses the bash notation of `$?` to call the exit value of the last executed command
# and to save it in a variable called `timeout_exit_status`.


if [ $timeout_exit_status -eq 124 ]; then
    tar -czf $checkpoint_tarball ./checkpoint ./runs_train ./wandb
    mv $checkpoint_tarball ..
    exit 85
fi

mv ./checkpoint ./checkpoint_cft
mv ./runs_train ./runs_train_cft
mv ./wandb ./wandb_cft

tar -czvf training_output_cft_${2}layers_seed${3}_rels.tar ./checkpoint_cft ./runs_train_cft
cp -r training_output_cft_*.tar /staging/gmerz2

rm -rf *

exit $timeout_exit_status
