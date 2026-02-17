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
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  data=lm1b-wrap \
  wandb.project=lm1b_full \
  wandb.name=lm1b_scratch_small_temp_0.1_ce_loss_no_grid_with_boundary_sampler_alpha_flow_ratio_0.5 \
  model=small \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=null \
  +trainer.check_val_every_n_epoch=1 \
  model.length=128 \
  optim.lr=3e-4 \
  algo=flm_shortcut \
  algo.flow_ratio=0.5 \
  algo.shortcut_loss_type=ce \
  algo.flow_warmup=False \
  algo.double_temb=True \
  algo.use_discrete_schedule=True \
  algo.gumbel_tau_log10_start=-1.0 \
  algo.gumbel_tau_log10_end=-1.0 \
  algo.curriculum_start=0 \
  algo.curriculum_end=25000 \
  algo.sample_d_on_grid=False \
  algo.use_continuous_shortcut=True \
  algo.add_boundary=True \
  sampling.noise_removal=shortcut_alpha \
  algo.scale_input=False \
  algo.scale_loss=False \
  checkpointing.resume_from_ckpt=False  

