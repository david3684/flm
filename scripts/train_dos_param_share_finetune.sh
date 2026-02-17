#!/bin/bash
#SBATCH -J dos-distill                   # Job name
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

# DOS_distill training command
# Teacher checkpoint path - update this to your actual teacher checkpoint
TEACHER_CKPT="YOUR_TEACHER_CHECKPOINT_PATH"

python -u -m main \
  loader.global_batch_size=512 \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  data=lm1b-wrap \
  data.cache_dir=YOUR_DATASET_PATH \
  wandb.project=dos_distill \
  wandb.name=shortcut_param_share_2_head_shortcut_half_on_gamma_with_loss_scale_detach_backbone\
  model=small \
  algo=flm_shortcut \
  training.loss_type=shortcut \
  training.finetune_path=$TEACHER_CKPT \
  trainer.max_steps=1000000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=5000 \
  model.length=128 \
  sampling.steps="[1,2,4,32]" \
  +sampling.noise_removal_list="[shortcut, shortcut_alpha]" \
  sampling.solver=euler \
  optim.lr=3e-4 \
  algo.shortcut_loss_type=mse \
  algo.double_temb=True \
  algo.use_discrete_schedule=True \
  algo.sample_d_on_grid=False \
  algo.use_continuous_shortcut=True \
  algo.shortcut_k_max=128 \
  algo.shortcut_on_alpha_t=False \
  algo.add_boundary=False \
  algo.scale_loss=False \
  algo.scale_input=False \
  algo.bootstrap_ema=True \
  algo.bootstrap_argmax=False \
  algo.n_separated_blocks=2 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=50000 \
  checkpointing.resume_from_ckpt=False

