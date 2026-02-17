EXP_NAME="SYNTHv2-tvm-flow-train-shorcut"

python -u -m main \
    loader.batch_size=128 \
    loader.global_batch_size=128 \
    loader.eval_batch_size=128 \
    data=synthetic-align \
    wandb.project=dos_distill_synth \
    wandb.name=$EXP_NAME \
    model=tiny \
    model.length=10 \
    algo=flm_shortcut_distill \
    optim.lr=3e-4 \
    data.vocab_size=1024 \
    trainer.log_every_n_steps=50 \
    trainer.max_steps=100000 \
    trainer.limit_val_batches=2 \
    trainer.val_check_interval=null \
    +trainer.check_val_every_n_epoch=5 \
    training.loss_type=lagrangian \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
    +sampling.noise_removal_list="[shortcut_alpha]" \
    eval.gen_ppl_eval_model_name_or_path=gpt2-large \
    sampling.steps="[1,2,4,8,128]" \
    algo.double_temb=False \
    algo.shortcut_loss_type=mse \
    algo.double_temb=True \
    algo.use_discrete_schedule=True \
    algo.sample_d_on_grid=False \
    algo.use_continuous_shortcut=True \
    algo.shortcut_k_max=128 \
    algo.shortcut_on_alpha_t=False \
    algo.add_boundary=False \
    algo.scale_loss=False \
    algo.scale_input=True \
    algo.bootstrap_ema=True \
    algo.bootstrap_argmax=False \
    algo.n_separated_blocks=-1 \
    algo.shortcut_scale_loss=True \
    algo.zero_center_residual=False \
    +algo.teacher_path="YOUR_TEACHER_CHECKPOINT_PATH" \
    checkpointing.resume_from_ckpt=False \
    # +wandb.offline=False
    
    