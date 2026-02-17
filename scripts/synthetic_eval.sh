STEPS=1

python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=tiny \
      model.length=15 \
      data=synthetic-align \
      algo=flm_shortcut_distill \
      +algo.teacher_path="YOUR_TEACHER_CHECKPOINT_PATH" \
      eval.checkpoint_path="YOUR_CHECKPOINT_PATH" \
      sampling.tau_log10=0.0 \
      loader.batch_size=2 \
      data.vocab_size=1024 \
      loader.eval_batch_size=32 \
      sampling.num_sample_batches=10 \
      sampling.steps=$STEPS \
      sampling.solver=euler \
      algo.double_temb=True \
      sampling.autoguidance_weight=0.0 \
      sampling.noise_removal=shortcut_alpha \
      eval.disable_ema=False \
      +wandb.offline=true \
