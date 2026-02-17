CKPT_PATH="YOUR_CHECKPOINT_PATH"
STEPS=32

export HYDRA_FULL_ERROR=1

python -u -m main \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=64 \
  data=lm1b-wrap \
  algo=mdlm \
  model=small \
  model.length=128 \
  eval.checkpoint_path=$CKPT_PATH \
  sampling.num_sample_batches=1 \
  sampling.steps=$STEPS \
  +wandb.offline=true \
  sampling.predictor=ancestral_cache \
  sampling.noise_removal=ancestral
