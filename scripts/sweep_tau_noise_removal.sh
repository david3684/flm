#!/bin/bash

# Sweep script for output_temperature and noise_removal parameters
# tau_log10 is fixed at -1.0

OUTPUT_TEMP_VALUES=(1.0 0.1 0.01 0.0)
NOISE_REMOVAL_VALUES=(uniform_alpha uniform)
TAU_LOG10=-1.0

CHECKPOINT_PATH="YOUR_CHECKPOINT_PATH"
# CHECKPOINT_PATH="YOUR_CHECKPOINT_PATH"
# CHECKPOINT_PATH="YOUR_CHECKPOINT_PATH"

LOG_FILE="temp_eval_tau_noise_0.01.log"
SUMMARY=""

echo "=========================================================="
echo " Starting evaluation sweep"
echo " Checkpoint: $CHECKPOINT_PATH"
echo " tau_log10: $TAU_LOG10 (fixed)"
echo " Sweeping output_temperature: ${OUTPUT_TEMP_VALUES[@]}"
echo " Sweeping noise_removal: ${NOISE_REMOVAL_VALUES[@]}"
echo "=========================================================="

for OUTPUT_TEMP in "${OUTPUT_TEMP_VALUES[@]}"
do
    for NOISE_REMOVAL in "${NOISE_REMOVAL_VALUES[@]}"
    do
        echo -e "\n[Running] sampling.output_temperature=$OUTPUT_TEMP, sampling.noise_removal=$NOISE_REMOVAL ..."
        echo "----------------------------------------------------------"

        # Set argmax_correction and actual temperature based on output_temperature
        if [ "$OUTPUT_TEMP" = "0.0" ]; then
            ARGMAX_CORRECTION="True"
            ACTUAL_TEMP=1.0
        else
            ARGMAX_CORRECTION="False"
            ACTUAL_TEMP=$OUTPUT_TEMP
        fi

        python -u -m main \
          mode=sample_eval \
          seed=1 \
          model=small \
          model.length=128 \
          data=lm1b-wrap \
          algo=flm \
          eval.checkpoint_path="$CHECKPOINT_PATH" \
          sampling.tau_log10=$TAU_LOG10 \
          loader.batch_size=2 \
          loader.eval_batch_size=32 \
          sampling.num_sample_batches=2 \
          sampling.steps=128 \
          sampling.solver=euler \
          algo.double_temb=False \
          +sampling.output_temperature=$ACTUAL_TEMP \
          sampling.noise_removal=$NOISE_REMOVAL \
          sampling.hard_start=False \
          sampling.argmax_correction=$ARGMAX_CORRECTION \
          algo.time_condition=alpha_t \
          algo.use_curriculum=True \
          +algo.n_separated_blocks=-1 \
          eval.disable_ema=False \
          +wandb.offline=true 2>&1 | tee $LOG_FILE

        PPL=$(sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" $LOG_FILE | grep -a "Generative perplexity:" | awk '{print $NF}')
        ENTROPY=$(sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" $LOG_FILE | grep -a "Sample entropy:" | awk '{print $NF}')

        if [ -z "$PPL" ]; then PPL="ERROR/NA"; fi
        if [ -z "$ENTROPY" ]; then ENTROPY="ERROR/NA"; fi

        SUMMARY+="$OUTPUT_TEMP\t$NOISE_REMOVAL\t$ARGMAX_CORRECTION\t$PPL\t$ENTROPY\n"
        
        echo "----------------------------------------------------------"
        echo ">> output_temperature=$OUTPUT_TEMP, noise_removal=$NOISE_REMOVAL, argmax_correction=$ARGMAX_CORRECTION: PPL=$PPL, Entropy=$ENTROPY"
    done
done

echo -e "\n=========================================================="
echo " FINAL EVALUATION SUMMARY"
echo "=========================================================="
printf "%-18s | %-15s | %-18s | %-20s | %-20s\n" "output_temperature" "noise_removal" "argmax_correction" "Gen PPL" "Sample Entropy"
echo "----------------------------------------------------------"
echo -e "$SUMMARY" | while read -r line; do
    if [ ! -z "$line" ]; then
        TEMP_VAL=$(echo "$line" | cut -f1)
        NOISE_VAL=$(echo "$line" | cut -f2)
        ARGMAX_VAL=$(echo "$line" | cut -f3)
        PP=$(echo "$line" | cut -f4)
        EN=$(echo "$line" | cut -f5)
        printf "%-18s | %-15s | %-18s | %-20s | %-20s\n" "$TEMP_VAL" "$NOISE_VAL" "$ARGMAX_VAL" "$PP" "$EN"
    fi
done
echo "=========================================================="

# Clean up temporary file
rm -f $LOG_FILE

