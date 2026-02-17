# CKPT_PATH="YOUR_CHECKPOINT_PATH"
CKPT_PATH="YOUR_CHECKPOINT_PATH"
STEPS=512

python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=small \
      model.length=1024 \
      data=openwebtext-split \
      algo=flm \
      eval.checkpoint_path=$CKPT_PATH \
      loader.batch_size=2 \
      loader.eval_batch_size=16 \
      sampling.num_sample_batches=64 \
      sampling.steps=$STEPS \
      sampling.solver=euler \
      algo.double_temb=False \
      sampling.noise_removal=uniform_alpha \
      algo.time_condition=alpha_t \
      eval.disable_ema=False \
      eval.ema_decay=0.9999 \
      +wandb.offline=true \
      
