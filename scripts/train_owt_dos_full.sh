#!/bin/bash
#SBATCH -J duo-lm1b                   # Job name
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

python -u -m main \
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=openwebtext-split \
  wandb.project=owt_full \
  wandb.name=owt_full_flow \
  model=small \
  algo=flm \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=20000 \
  model.length=1024 \
  optim.lr=3e-4 \
  algo.flow_warmup=True \
  algo.double_temb=False \
  algo.use_discrete_schedule=True \
  algo.gumbel_tau_log10_start=-1.0 \
  algo.gumbel_tau_log10_end=-1.0 \
  algo.curriculum_start=0 \
  algo.curriculum_end=100000 \
  sampling.noise_removal=uniform \
  algo.time_condition=alpha_t \
  algo.scale_input=False \
  algo.scale_loss=False \
  algo.flow_loss_type=mse \
  checkpointing.resume_from_ckpt=False


