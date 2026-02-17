#!/bin/bash

TEACHER_CKPT_PATH="YOUR_CHECKPOINT_PATH"
# CKPT_PATH="YOUR_CHECKPOINT_PATH"
CKPT_PATH="YOUR_CHECKPOINT_PATH" "

if [ $# -eq 0 ]; then
    STEPS_LIST=(128)
else
    STEPS_LIST=("$@")
fi
SUMMARY=""

LOG_FILE="temp_eval.log"

# 결과 JSON 파일들을 모을 폴더 생성
RESULTS_DIR="double_distill_100k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "Results will be collected in: $RESULTS_DIR"

echo "=========================================================="
echo " Starting evaluation sweep"
echo " Checkpoint: $CKPT_PATH"
echo "=========================================================="

for STEPS in "${STEPS_LIST[@]}"
do
    echo -e "\n[Running] sampling.steps=$STEPS ..."
    echo "----------------------------------------------------------"

    python -u -m main \
        mode=sample_eval \
        seed=578\
        model=small \
        model.length=128 \
        data=lm1b-wrap \
        algo=flm_shortcut_distill_double \
        +algo.teacher_path=$TEACHER_CKPT_PATH \
        eval.checkpoint_path=$CKPT_PATH \
        sampling.tau_log10=0.0 \
        loader.batch_size=2 \
        loader.eval_batch_size=2 \
        sampling.num_sample_batches=1 \
        sampling.steps=$STEPS \
        sampling.solver=euler \
        algo.double_temb=True \
        sampling.noise_removal=shortcut_alpha \
        algo.learnable_loss_weighting=True \
        eval.disable_ema=False \
        algo.n_separated_blocks=-1 \
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
