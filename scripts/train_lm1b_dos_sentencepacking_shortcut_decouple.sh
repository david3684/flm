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
  loader.eval_batch_size=128 \
  data=lm1b-wrap \
  wandb.project=lm1b_full \
  wandb.name=fm_0.01_sc_0.01_scratch_no_grid_w_boundary_decouple_4_blocks \
  model=small \
  training.loss_type=shortcut \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=10000 \
  model.length=128 \
  optim.lr=3e-4 \
  algo=flm_shortcut \
  algo.flow_ratio=0.75 \
  algo.shortcut_loss_type=mse \
  algo.flow_warmup=False \
  algo.double_temb=True \
  algo.use_discrete_schedule=True \
  algo.sample_d_on_grid=False \
  algo.use_continuous_shortcut=True \
  algo.add_boundary=True \
  algo.tau_log10_fm=-2.0 \
  algo.tau_log10_shortcut=-2.0 \
  sampling.tau_log10=-1.0 \
  sampling.solver=DPMv2 \
  sampling.noise_removal=shortcut \
  algo.scale_loss=False \
  algo.scale_input=False \
  algo.bootstrap_ema=True \
  algo.bootstrap_argmax=False \
  algo.shortcut_k_max=128 \
  algo.n_separated_blocks=4 \
  checkpointing.resume_from_ckpt=True \
  checkpointing.resume_ckpt_path="YOUR_RESUME_CHECKPOINT_PATH" \
  wandb.id=f0af8df1