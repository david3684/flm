CKPT_PATH="YOUR_CHECKPOINT_PATH"
STEPS=4

export HYDRA_FULL_ERROR=1

python -u -m main \
  is_di4c=False \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  algo=duo_base \
  model=small \
  model.length=1024 \
  eval.checkpoint_path=$CKPT_PATH \
  sampling.num_sample_batches=64 \
  sampling.steps=$STEPS \
  +wandb.offline=true \
  sampling.noise_removal=ancestral \
  sampling.predictor=ancestral
