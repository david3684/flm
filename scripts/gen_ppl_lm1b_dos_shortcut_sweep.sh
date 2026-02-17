#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./sweep_steps.sh [CHECKPOINT_PATH] [STEPS...]"
    echo "Example: ./sweep_steps.sh /path/to/checkpoint 16 32 64"
    exit 1
fi

CHECKPOINT_PATH=$1
shift  

if [ $# -eq 0 ]; then
    STEPS_LIST=(128)
else
    STEPS_LIST=("$@")
fi
SUMMARY=""

LOG_FILE="temp_eval.log"

echo "=========================================================="
echo " Starting evaluation sweep"
echo " Checkpoint: $CHECKPOINT_PATH"
echo "=========================================================="

for STEPS in "${STEPS_LIST[@]}"
do
    echo -e "\n[Running] sampling.steps=$STEPS ..."
    echo "----------------------------------------------------------"

    python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=small \
      model.length=128 \
      data=lm1b-wrap \
      algo=flm_shortcut \
      eval.checkpoint_path="$CHECKPOINT_PATH" \
      sampling.tau_log10=-1.0 \
      loader.batch_size=2 \
      loader.eval_batch_size=32 \
      sampling.num_sample_batches=4 \
      sampling.steps=$STEPS \
      sampling.solver=DPMv2 \
      algo.flow_warmup=False \
      algo.double_temb=True \
      sampling.noise_removal=shortcut_alpha\
      sampling.hard_start=False \
      sampling.argmax_correction=False \
      algo.shortcut_on_alpha_t=False \
      algo.use_curriculum=True \
      algo.scale_input=False \
      algo.n_separated_blocks=2 \
      eval.disable_ema=False \
      +wandb.offline=true 2>&1 | tee $LOG_FILE

    PPL=$(sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" $LOG_FILE | grep -a "Generative perplexity:" | awk '{print $NF}')
    ENTROPY=$(sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" $LOG_FILE | grep -a "Sample entropy:" | awk '{print $NF}')

    if [ -z "$PPL" ]; then PPL="ERROR/NA"; fi
    if [ -z "$ENTROPY" ]; then ENTROPY="ERROR/NA"; fi

    SUMMARY+="$STEPS\t$PPL\t$ENTROPY\n"
    
    echo "----------------------------------------------------------"
    echo ">> Steps $STEPS Result: PPL=$PPL, Entropy=$ENTROPY"
done

echo -e "\n=========================================================="
echo " FINAL EVALUATION SUMMARY"
echo "=========================================================="
printf "%-10s | %-20s | %-20s\n" "Steps" "Gen PPL" "Sample Entropy"
echo "----------------------------------------------------------"
echo -e "$SUMMARY" | while read -r line; do
    if [ ! -z "$line" ]; then
        ST=$(echo "$line" | cut -f1)
        PP=$(echo "$line" | cut -f2)
        EN=$(echo "$line" | cut -f3)
        printf "%-10s | %-20s | %-20s\n" "$ST" "$PP" "$EN"
    fi
done
echo "=========================================================="

# 임시 파일 삭제
rm $LOG_FILE