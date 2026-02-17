EXP_NAME="SYNTHv2-tvm-flow-train"

python -u -m main \
    loader.batch_size=32 \
    loader.global_batch_size=128 \
    loader.eval_batch_size=128 \
    data=synthetic-align \
    wandb.project=dos_distill_synth \
    wandb.name=$EXP_NAME \
    model=tiny \
    model.length=10 \
    algo=flm \
    training.loss_type=flow \
    trainer.log_every_n_steps=50 \
    trainer.max_steps=100000 \
    data.vocab_size=1024 \
    trainer.limit_val_batches=2 \
    trainer.val_check_interval=null \
    +trainer.check_val_every_n_epoch=20 \
    algo.flow_loss_type=ce \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
    +sampling.noise_removal_list="[uniform_alpha]" \
    eval.gen_ppl_eval_model_name_or_path=gpt2-large \
    sampling.steps="[1,2,4,8,128]" \
    algo.double_temb=False \