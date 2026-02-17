#!/bin/bash
#SBATCH -J duo-base                   # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=anonymous          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.


# Assuming the finetune_path corresponds to the DUO model
# trained for 500K steps with curriculum learning, we train the
# model for 500K more steps.
python -u -m main \
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=openwebtext-split \
  data.cache_dir=YOUR_DATASET_PATH \
  wandb.name=owt_full_ce_loss \
  model=small \
  algo=flm \
  model.length=1024 \
  training.loss_type=flow \
  sampling.num_sample_batches=1 \
  sampling.noise_removal=uniform-alpha \
  sampling.solver=euler \
  sampling.steps=[1024] \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  optim.lr=3e-4 \
  trainer.val_check_interval=5000 \
  algo.double_temb=False \
  algo.use_discrete_schedule=True \
  algo.time_condition=alpha_t \
  algo.loss_scale=none \
  algo.gumbel_tau_log10_start=-0.0 \
  algo.gumbel_tau_log10_end=-0.0 \
  algo.curriculum_start=0 \
  algo.curriculum_end=100000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
