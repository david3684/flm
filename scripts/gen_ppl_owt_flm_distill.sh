#!/bin/bash

TEACHER_CKPT_PATH="YOUR_CHECKPOINT_PATH"
CKPT_PATH="YOUR_CHECKPOINT_PATH"
STEPS=32

python -u -m main \
    mode=sample_eval \
    seed=1 \
    model=small \
    model.length=1024 \
    data=lm1b-wrap \
    algo=flm_shortcut_distill \
    algo.teacher_path=$TEACHER_CKPT_PATH \
    eval.checkpoint_path=$CKPT_PATH \
    loader.batch_size=2 \
    loader.eval_batch_size=1 \
    sampling.num_sample_batches=8\
    sampling.steps=$STEPS \
    algo.double_temb=True \
    algo.learnable_loss_weighting=True \
    eval.disable_ema=False \
    +wandb.offline=true
