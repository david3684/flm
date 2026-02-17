
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

LOG_FILE="temp_eval_2.log"

# 결과 JSON 파일들을 모을 폴더 생성
RESULTS_DIR="sweep_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "Results will be collected in: $RESULTS_DIR"

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
      model.length=1024 \
      data=openwebtext-split \
      algo=flm \
      eval.checkpoint_path="$CHECKPOINT_PATH" \
      sampling.tau_log10=0.0 \
      loader.batch_size=2 \
      loader.eval_batch_size=16 \
      sampling.num_sample_batches=64 \
      sampling.steps=$STEPS \
      sampling.solver=euler \
      algo.double_temb=False \
      sampling.noise_removal=uniform_alpha \
      algo.time_condition=alpha_t \
      eval.disable_ema=False \
      +wandb.offline=true 2>&1 | tee $LOG_FILE

    PPL=$(sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" $LOG_FILE | grep -a "Generative perplexity:" | awk '{print $NF}')
    ENTROPY=$(sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" $LOG_FILE | grep -a "Sample entropy:" | awk '{print $NF}')
    
    # JSON 파일 경로 추출 및 복사
    JSON_PATH=$(sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" $LOG_FILE | grep -a "Samples saved at:" | awk '{print $NF}')
    if [ ! -z "$JSON_PATH" ] && [ -f "$JSON_PATH" ]; then
        cp "$JSON_PATH" "$RESULTS_DIR/samples_${STEPS}.json"
        echo ">> Copied JSON to: $RESULTS_DIR/samples_${STEPS}.json"
    fi

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
echo ""
echo "All JSON files collected in: $RESULTS_DIR"
echo "Files:"
ls -lh "$RESULTS_DIR"/*.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

# 임시 파일 삭제
rm $LOG_FILE

