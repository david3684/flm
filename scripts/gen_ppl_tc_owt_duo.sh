#!/bin/bash
#SBATCH -J an_owt_duo                    # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=16000                   # server memory requested (per node)
#SBATCH -t 24:00:00                   # Time limit (hh:mm:ss)
#SBATCH --partition=anonymous,gpu      # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

export HYDRA_FULL_ERROR=1

checkpoint_path="CKPT_PATH"
ckpt=duo-distilled

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --steps) steps="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --ckpt) ckpt="$2"; shift ;;
        --checkpoint_path) checkpoint_path="$2"; shift ;;
        --temperature) temperature="$2"; shift ;;
        --disable_ema) disable_ema="$2"; shift ;;
        --duplicate) duplicate="$2"; shift ;;
        --num_sample_batches) num_sample_batches="$2"; shift ;;
        --eval_batch_size) eval_batch_size="$2"; shift ;;
        --sampling) sampling="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

steps=${steps:-1024}
seed=${seed:-42}
temperature=${temperature:-1.0}
num_sample_batches=${num_sample_batches:-128}
disable_ema=${disable_ema:-False}
duplicate=${duplicate:-8}
sampling=${sampling:-"greedy"}
eval_batch_size=${eval_batch_size:-8}

generated_samples_path=$checkpoint_path/samples_ancestral_greedy/$seed-$steps-$num_sample_batches-$eval_batch_size-$ckpt-$temperature-disable-ema-$disable_ema-$duplicate-llama3_1.json

# for steps in 2 4 8 16 32 1024 64 128 256 512; do
# for steps in 2 4 8 16 32 1024; do
# for steps in 1024; do
echo "  Steps: $steps"
echo "  Seed: $seed"
echo "  ckpt: $ckpt"
echo "  temperature: $temperature"
echo "  num_sample_batches: $num_sample_batches"
echo "  eval_batch_size: $eval_batch_size"
echo "  sampling: $sampling"
echo "  generated_samples_path: $generated_samples_path"

# sample_eval_with_tc \

python -u -m main \
    mode=sample_eval_with_tc \
    seed=$seed \
    loader.batch_size=2 \
    loader.eval_batch_size=$eval_batch_size \
    data=openwebtext-split \
    algo=duo_base \
    model=small \
    eval.checkpoint_path=$checkpoint_path/$ckpt.ckpt \
    sampling.num_sample_batches=$num_sample_batches \
    sampling.steps=$steps \
    sampling.predictor=ancestral \
    +sampling.duplicate=$duplicate \
    +wandb.offline=true \
    eval.generated_samples_path=$generated_samples_path \
    sampling.temperature=$temperature \
    sampling.noise_removal=$sampling \
    eval.disable_ema=$disable_ema