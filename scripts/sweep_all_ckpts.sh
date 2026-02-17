#!/bin/bash

TEACHER_CKPT_PATH="YOUR_CHECKPOINT_PATH"

# Usage check
if [ $# -lt 2 ]; then
    echo "Usage: $0 <checkpoint_folder> <steps1> [steps2] [steps3] ..."
    echo "Example: $0 /path/to/checkpoints 64 128 256"
    exit 1
fi

CKPT_FOLDER="$1"
shift
STEPS_LIST=("$@")

# 폴더 존재 확인
if [ ! -d "$CKPT_FOLDER" ]; then
    echo "Error: Folder $CKPT_FOLDER does not exist"
    exit 1
fi

# 결과를 저장할 메인 폴더
MAIN_RESULTS_DIR="sweep_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MAIN_RESULTS_DIR"

LOG_FILE="temp_eval.log"
SUMMARY_FILE="$MAIN_RESULTS_DIR/summary.txt"

echo "=========================================================="
echo " Checkpoint Sweep Evaluation"
echo " Folder: $CKPT_FOLDER"
echo " Steps: ${STEPS_LIST[@]}"
echo " Results will be saved in: $MAIN_RESULTS_DIR"
echo "=========================================================="

# .ckpt 파일 목록 가져오기 (정렬)
CKPT_FILES=($(find "$CKPT_FOLDER" -maxdepth 1 -name "*.ckpt" -type f | sort))

if [ ${#CKPT_FILES[@]} -eq 0 ]; then
    echo "Error: No .ckpt files found in $CKPT_FOLDER"
    exit 1
fi

echo "Found ${#CKPT_FILES[@]} checkpoint files"
echo ""

# Summary 헤더
echo "=========================================================" > "$SUMMARY_FILE"
echo " Checkpoint Sweep Summary" >> "$SUMMARY_FILE"
echo " Date: $(date)" >> "$SUMMARY_FILE"
echo "=========================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 각 체크포인트에 대해 반복
for CKPT_PATH in "${CKPT_FILES[@]}"
do
    CKPT_NAME=$(basename "$CKPT_PATH")
    echo ""
    echo "=========================================================="
    echo " Processing: $CKPT_NAME"
    echo "=========================================================="
    
    # 이 체크포인트의 결과를 저장할 폴더
    CKPT_RESULTS_DIR="$MAIN_RESULTS_DIR/${CKPT_NAME%.ckpt}"
    mkdir -p "$CKPT_RESULTS_DIR"
    
    CKPT_SUMMARY=""
    
    for STEPS in "${STEPS_LIST[@]}"
    do
        echo -e "\n[Running] $CKPT_NAME with sampling.steps=$STEPS ..."
        echo "----------------------------------------------------------"
        
        python -u -m main \
            mode=sample_eval \
            seed=3 \
            model=small \
            model.length=128 \
            data=lm1b-wrap \
            algo=flm_shortcut_distill_double \
            +algo.teacher_path=$TEACHER_CKPT_PATH \
            eval.checkpoint_path=$CKPT_PATH \
            sampling.tau_log10=0.0 \
            loader.batch_size=2 \
            loader.eval_batch_size=128 \
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
            cp "$JSON_PATH" "$CKPT_RESULTS_DIR/samples_steps${STEPS}.json"
            echo ">> Copied JSON to: $CKPT_RESULTS_DIR/samples_steps${STEPS}.json"
        fi
        
        if [ -z "$PPL" ]; then PPL="ERROR/NA"; fi
        if [ -z "$ENTROPY" ]; then ENTROPY="ERROR/NA"; fi
        
        CKPT_SUMMARY+="$STEPS\t$PPL\t$ENTROPY\n"
        
        echo "----------------------------------------------------------"
        echo ">> $CKPT_NAME @ Steps $STEPS: PPL=$PPL, Entropy=$ENTROPY"
    done
    
    # 이 체크포인트의 요약을 메인 summary 파일에 추가
    echo "Checkpoint: $CKPT_NAME" >> "$SUMMARY_FILE"
    echo "Path: $CKPT_PATH" >> "$SUMMARY_FILE"
    printf "%-10s | %-20s | %-20s\n" "Steps" "Gen PPL" "Sample Entropy" >> "$SUMMARY_FILE"
    echo "----------------------------------------------------------" >> "$SUMMARY_FILE"
    echo -e "$CKPT_SUMMARY" | while read -r line; do
        if [ ! -z "$line" ]; then
            ST=$(echo "$line" | cut -f1)
            PP=$(echo "$line" | cut -f2)
            EN=$(echo "$line" | cut -f3)
            printf "%-10s | %-20s | %-20s\n" "$ST" "$PP" "$EN" >> "$SUMMARY_FILE"
        fi
    done
    echo "" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # 콘솔 출력
    echo -e "\n----------------------------------------------------------"
    echo " Summary for $CKPT_NAME"
    echo "----------------------------------------------------------"
    printf "%-10s | %-20s | %-20s\n" "Steps" "Gen PPL" "Sample Entropy"
    echo "----------------------------------------------------------"
    echo -e "$CKPT_SUMMARY" | while read -r line; do
        if [ ! -z "$line" ]; then
            ST=$(echo "$line" | cut -f1)
            PP=$(echo "$line" | cut -f2)
            EN=$(echo "$line" | cut -f3)
            printf "%-10s | %-20s | %-20s\n" "$ST" "$PP" "$EN"
        fi
    done
done

echo ""
echo "=========================================================="
echo " ALL CHECKPOINTS COMPLETED"
echo "=========================================================="
echo ""
echo "Results saved in: $MAIN_RESULTS_DIR"
echo "Summary file: $SUMMARY_FILE"
echo ""
echo "Directory structure:"
tree -L 2 "$MAIN_RESULTS_DIR" 2>/dev/null || find "$MAIN_RESULTS_DIR" -type f

# 임시 파일 삭제
rm -f $LOG_FILE

echo ""
echo "To view summary:"
echo "  cat $SUMMARY_FILE"
