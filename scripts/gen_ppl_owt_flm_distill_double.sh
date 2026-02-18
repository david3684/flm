CKPT_PATH="YOUR_CHECKPOINT_PATH"
STEPS=32

python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=small \
      model.length=1024 \
      data=openwebtext-split \
      algo=flm_distill_double \
      eval.checkpoint_path=$CKPT_PATH \
      loader.batch_size=2 \
      loader.eval_batch_size=1 \
      sampling.num_sample_batches=1 \
      sampling.steps=$STEPS \
      algo.double_temb=True \
      eval.disable_ema=False \
      +wandb.offline=true \
