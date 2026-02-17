# CKPT_PATH="YOUR_CHECKPOINT_PATH"
CKPT_PATH="YOUR_CHECKPOINT_PATH"
STEPS=512

python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=small \
      model.length=128 \
      data=lm1b-wrap \
      algo=flm \
      eval.checkpoint_path=$CKPT_PATH \
      sampling.tau_log10=0.0 \
      loader.batch_size=2 \
      loader.eval_batch_size=1 \
      sampling.num_sample_batches=1 \
      sampling.steps=$STEPS \
      sampling.solver=euler \
      algo.double_temb=False \
      sampling.autoguidance_weight=0.0 \
      sampling.noise_removal=uniform_alpha \
      algo.time_condition=alpha_t \
      eval.disable_ema=False \
      +wandb.offline=true \