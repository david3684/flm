DATA_DIR="YOUR_DATA_DIR"

python -u -m main \
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  wandb.project=lm1b_full \
  wandb.name=lm1b_full_flm \
  model=small \
  algo=flm \
  model.length=128 \
  sampling.num_sample_batches=1 \
  sampling.steps=[1024] \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  optim.lr=3e-4 \
  trainer.val_check_interval=5000 \
  algo.double_temb=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
