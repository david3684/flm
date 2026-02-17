CKPT_PATH="YOUR_CHECKPOINT_PATH"

python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=small \
      model.length=1024 \
      data=openwebtext-split \
      algo=flm_shortcut \
      eval.checkpoint_path=$CKPT_PATH \
      sampling.tau_log10=0.0 \
      loader.batch_size=2 \
      loader.eval_batch_size=32 \
      sampling.num_sample_batches=1 \
      sampling.steps=32 \
      sampling.solver=euler \
      algo.double_temb=True \
      sampling.noise_removal=shortcut_alpha \
      algo.time_condition=alpha_t \
      eval.disable_ema=False \
      algo.n_separated_blocks=-1 \
      +wandb.offline=true


