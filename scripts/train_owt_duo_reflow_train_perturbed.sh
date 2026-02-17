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

export HYDRA_FULL_ERROR=1

checkpoint_path="CKPT_PATH"
dataset_path="DATASET_PATH"
ckpt=duo-distilled

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ckpt) ckpt="$2"; shift ;;
        --checkpoint_path) checkpoint_path="$2"; shift ;;
        --dataset_path) dataset_path="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

#torchrun --standalone --nproc_per_node= main.py \
python -u -m main \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split-reflow \
  data.cache_dir=$dataset_path \
  wandb.name=ReDi_train \
  model=small \
  algo=rectification \
  model.length=1024 \
  sampling.steps=32 \
  algo.gamma_min=-3.55 \
  algo.gamma_max=-1.85 \
  training.finetune_path=$checkpoint_path/$ckpt.ckpt \
  training.not_sampling_t=true \
  trainer.val_check_interval=null \
  trainer.max_steps=1000000