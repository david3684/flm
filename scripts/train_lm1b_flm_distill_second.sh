DATA_DIR="YOUR_DATA_DIR"
TEACHER_F_PATH="YOUR_FLM_CHECKPOINT_PATH"
TEACHER_G_PATH="YOUR_FMLM_CHECKPOINT_PATH"

python -u -m main \
    loader.global_batch_size=512 \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    data=lm1b-wrap \
    data.cache_dir=$DATA_DIR \
    wandb.project=flm_distill \
    wandb.name=second_phase_distill \
    model=small \
    algo=flm_distill_double \
    algo.teacher_f_path=$TEACHER_F_PATH \
    algo.teacher_g_path=$TEACHER_G_PATH \
    trainer.max_steps=1000000 \
    trainer.precision=bf16 \
    trainer.val_check_interval=5000 \
    model.length=128 \
    sampling.steps=[1,2,4,32] \
    optim.lr=3e-4 \
    algo.double_temb=True \
    algo.add_boundary=True \
    +algo.boundary_prob=128 \
    algo.learnable_loss_weighting=True \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
