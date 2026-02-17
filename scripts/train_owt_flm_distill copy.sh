python -u -m main \
    loader.global_batch_size=512 \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    data=openwebtext-split \
    data.cache_dir=/home/work/RADAR/workspace/KAIST/cdBDD/datasets/openwebtext \
    wandb.project=owt_distill \
    wandb.name=flm_diatill \
    model=small \
    algo=flm_semigroup_distill \
    algo.teacher_path=/home/work/RADAR/workspace/KAIST/discrete-mean-flow/text/outputs/lm1b/2026.01.07/000819/checkpoints/73-1000000.ckpt \
    trainer.max_steps=1000000 \
    trainer.precision=bf16 \
    trainer.val_check_interval=10000 \
    model.length=1024 \
    sampling.steps=[1,2,4,32] \
    sampling.solver=euler \
    optim.lr=3e-4 \
    algo.double_temb=True \
    algo.add_boundary=True \
    +algo.boundary_prob=64 \
    algo.bootstrap_ema=False \
    algo.learnable_loss_weighting=True \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
    checkpointing.resume_from_ckpt=False \
    
