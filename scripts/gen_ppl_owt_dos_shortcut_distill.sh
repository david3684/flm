#!/bin/bash

TEACHER_CKPT_PATH="YOUR_CHECKPOINT_PATH"
# CKPT_PATH="YOUR_CHECKPOINT_PATH"
CKPT_PATH="YOUR_CHECKPOINT_PATH"
STEPS=512

python -u -m main \
    mode=sample_eval \
    seed=1 \
    model=small \
    model.length=1024 \
    data=openwebtext-split \
    algo=flm_shortcut_distill \
    +algo.teacher_path=$TEACHER_CKPT_PATH \
    eval.checkpoint_path=$CKPT_PATH \
    sampling.tau_log10=0.0 \
    loader.batch_size=2 \
    loader.eval_batch_size=8 \
    sampling.num_sample_batches=1 \
    sampling.steps=$STEPS \
    sampling.solver=euler \
    algo.double_temb=True \
    sampling.noise_removal=shortcut_alpha \
    +algo.freeze_token_embeddings=False \
    algo.learnable_loss_weighting=True \
    eval.disable_ema=False \
    algo.n_separated_blocks=-1 \
    +algo.base_num_steps=1 \
    +algo.progressive_distill=True \
    +algo.d_max_step=512 \
    +wandb.offline=true \
    # +algo.final_layer_remove_bias=True \
    # +algo.final_layer_weight_decay=True \
