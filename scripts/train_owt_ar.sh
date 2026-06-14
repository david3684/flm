#!/bin/bash

DATA_DIR="${DATA_DIR:-YOUR_DATA_DIR}"

python -u -m main \
  loader.global_batch_size=512 \
  loader.batch_size=512 \
  loader.eval_batch_size=128 \
  data=openwebtext-split \
  data.cache_dir=$DATA_DIR \
  wandb.project=flm \
  wandb.name=owt_full_ar \
  model=small \
  algo=ar \
  model.length=128 \
  sampling.num_sample_batches=1 \
  sampling.steps=[1024] \
  trainer.devices=1 \
  trainer.max_steps=100000 \
  trainer.precision=bf16 \
  optim.lr=3e-4 \
  trainer.val_check_interval=5000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000
