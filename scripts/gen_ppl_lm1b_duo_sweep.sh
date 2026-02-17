#!/bin/bash
#SBATCH -J sample_ar                  # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

if [ $# -eq 0 ]; then
    echo "Usage: ./gen_ppl_lm1b_duo_sweep.sh [STEPS...]"
    echo "Example: ./gen_ppl_lm1b_duo_sweep.sh 1 2 4 8 16"
    exit 1
fi

CHECKPOINT_PATH="YOUR_CHECKPOINT_PATH"
STEPS_LIST=("$@")
SUMMARY=""
SAMPLE_PATHS=()  # Array to store all sample JSON paths
SAMPLE_STEPS=()  # Array to store corresponding STEPS for each sample path

LOG_FILE="temp_eval.log"

echo "=========================================================="
echo " Starting evaluation sweep"
echo " Checkpoint: $CHECKPOINT_PATH"
echo "=========================================================="

export HYDRA_FULL_ERROR=1

for STEPS in "${STEPS_LIST[@]}"
do
    echo -e "\n[Running] sampling.steps=$STEPS ..."
    echo "----------------------------------------------------------"

    python -u -m main \
      mode=sample_eval \
      loader.batch_size=2 \
      loader.eval_batch_size=128 \
      data=lm1b-wrap \
      algo=duo_base \
      model=small \
      model.length=128 \
      eval.checkpoint_path="$CHECKPOINT_PATH" \
      sampling.num_sample_batches=1 \
      sampling.steps=$STEPS \
      +wandb.offline=true \
      sampling.noise_removal=ancestral \
      sampling.predictor=ancestral 2>&1 | tee $LOG_FILE

    PPL=$(sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" $LOG_FILE | grep -a "Generative perplexity:" | awk '{print $NF}')
    ENTROPY=$(sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" $LOG_FILE | grep -a "Sample entropy:" | awk '{print $NF}')
    SAMPLE_PATH=$(sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" $LOG_FILE | grep -a "Samples saved at:" | tail -1 | awk '{print $NF}')

    if [ -z "$PPL" ]; then PPL="ERROR/NA"; fi
    if [ -z "$ENTROPY" ]; then ENTROPY="ERROR/NA"; fi
    if [ ! -z "$SAMPLE_PATH" ] && [ -f "$SAMPLE_PATH" ]; then
        SAMPLE_PATHS+=("$SAMPLE_PATH")
        SAMPLE_STEPS+=("$STEPS")
    fi

    SUMMARY+="$STEPS\t$PPL\t$ENTROPY\n"

    echo "----------------------------------------------------------"
    echo ">> Steps $STEPS Result: PPL=$PPL, Entropy=$ENTROPY"
    if [ ! -z "$SAMPLE_PATH" ]; then
        echo "   Sample JSON: $SAMPLE_PATH"
    fi
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

# Print all sample JSON paths
if [ ${#SAMPLE_PATHS[@]} -gt 0 ]; then
    echo -e "\n=========================================================="
    echo " GENERATED SAMPLE JSON FILES"
    echo "=========================================================="
    for i in "${!SAMPLE_PATHS[@]}"; do
        echo "[$((i+1))] ${SAMPLE_PATHS[$i]}"
    done
    echo "=========================================================="
    
    # Copy all files to a single directory (enabled by default)
    # Set COLLECT_SAMPLES=0 to disable, or COLLECT_SAMPLES_DIR=/path/to/dir to specify directory
    if [ "${COLLECT_SAMPLES:-1}" != "0" ] || [ ! -z "$COLLECT_SAMPLES_DIR" ]; then
        if [ ! -z "$COLLECT_SAMPLES_DIR" ]; then
            COLLECT_DIR="$COLLECT_SAMPLES_DIR"
        else
            COLLECT_DIR="collected_samples_$(date +%Y%m%d_%H%M%S)"
        fi
        mkdir -p "$COLLECT_DIR"
        echo "Copying all sample JSONs to: $COLLECT_DIR"
        for i in "${!SAMPLE_PATHS[@]}"; do
            path="${SAMPLE_PATHS[$i]}"
            steps="${SAMPLE_STEPS[$i]}"
            if [ -f "$path" ]; then
                # Get file extension from original filename
                extension="${path##*.}"
                new_filename="sample_${steps}.${extension}"
                cp "$path" "$COLLECT_DIR/$new_filename"
                echo "  Copied: $new_filename"
            fi
        done
        echo "All samples copied to: $COLLECT_DIR"
    fi
fi

# 임시 파일 삭제
rm $LOG_FILE
