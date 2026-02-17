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
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  data=synthetic-align \
  wandb.project=lm1b_full \
  wandb.name=synthetic_shortcut_finetune_no_grid_0.1_cond_gamma_sampler_alpha_ce_loss\
  model=synthetic \
  algo=flm_shortcut_finetune \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=null \
  +trainer.check_val_every_n_epoch=50 \
  model.length=10 \
  optim.lr=3e-4 \
  algo.flow_warmup=False \
  algo.double_temb=True \
  sampling.noise_removal=shortcut_alpha \
  algo.shortcut_loss_type=ce \
  algo.gumbel_tau_log10_start=-1.0 \
  algo.gumbel_tau_log10_end=-1.0 \
  algo.shortcut_on_alpha_t=True \
  algo.sample_d_on_grid=True \
  algo.use_continuous_shortcut=False \
  algo.bootstrap_ema=True \
  algo.use_curriculum=True \
  algo.scale_input=False \
  eval.disable_ema=False \
  training.finetune_path="YOUR_FINETUNE_CHECKPOINT_PATH" \
  +wandb.offline=false



