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
    algo=duo_meanflow \
    seed=42 \
    wandb.project=dos_debug \
    wandb.name=owt_small_full_meanflow_scratch \
    data=openwebtext-split \
    eval.compute_generative_perplexity=True \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    trainer.max_steps=1000000 \
    trainer.log_every_n_steps=10 \
    trainer.val_check_interval=60000 \
    model=small \
    model.length=1024 \
    optim.lr=3e-4 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=50000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    training.loss_type=meanflow \
    +wandb.offline=False
