#!/bin/bash

TEACHER_CKPT_PATH="YOUR_CHECKPOINT_PATH"
# CKPT_PATH="YOUR_CHECKPOINT_PATH"
CKPT_PATH="YOUR_CHECKPOINT_PATH"
STEPS=1

python -u -m main \
    mode=sample_eval \
    seed=1 \
    model=small \
    model.length=128 \
    data=lm1b-wrap \
    algo=flm_shortcut_distill \
    +algo.teacher_path=$TEACHER_CKPT_PATH \
    eval.checkpoint_path=$CKPT_PATH \
    sampling.tau_log10=0.0 \
    loader.batch_size=2 \
    loader.eval_batch_size=1 \
    sampling.num_sample_batches=8\
    sampling.steps=$STEPS \
    sampling.solver=euler \
    algo.double_temb=True \
    sampling.noise_removal=shortcut_alpha \
    sampling.autoguidance_weight=0.0 \
    +sampling.parallel_forward=False \
    algo.learnable_loss_weighting=True \
    eval.disable_ema=False \
    algo.n_separated_blocks=-1 \
    +wandb.offline=true
