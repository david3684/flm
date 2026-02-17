#!/bin/bash
#SBATCH -J owt_duo_anneal                    # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=anonymous          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

checkpoint_path="${1:-CKPT_PATH}"
num_steps="${2:-10}"
seed="${3}"
if [ -z "$checkpoint_path" ] || [ -z "$num_steps" ] || [ -z "$seed" ]; then
    echo "Usage: $0 <checkpoint_path> <num_steps> <seed>"
    exit 1
fi

export HYDRA_FULL_ERROR=1

python -u -m main \
  mode=sample_eval \
  seed=$seed \
  model=small \
  algo=flm_shortcut \
  algo.use_curriculum=True \
  eval.checkpoint_path=$checkpoint_path \
  loader.batch_size=2 \
  loader.eval_batch_size=4 \
  sampling.num_sample_batches=8 \
  sampling.steps=$num_steps \
  sampling.noise_removal=shortcut \
  algo.flow_warmup=False \
  eval.disable_ema=False \
  algo.double_temb=False \
  algo.shortcut_on_alpha_t=True \
  +wandb.offline=true

