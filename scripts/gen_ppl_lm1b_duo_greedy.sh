CKPT_PATH="YOUR_CHECKPOINT_PATH" 
STEPS=512

export HYDRA_FULL_ERROR=1

python -u -m main \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=32 \
  data=lm1b-wrap \
  algo=duo_base \
  model=small \
  model.length=128 \
  eval.checkpoint_path=$CKPT_PATH \
  sampling.num_sample_batches=1 \
  sampling.steps=$STEPS \
  +wandb.offline=true \
  sampling.noise_removal=greedy \
  sampling.predictor=ancestral
