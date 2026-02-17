#!/bin/bash
#SBATCH -J owt_dos_shortcut_sweep          # Job name
#SBATCH -o watch_folder/%x_%j.out          # log file (out & err)
#SBATCH -N 1                               # Total number of nodes requested
#SBATCH --get-user-env                     # retrieve the users login environment
#SBATCH --mem=100000                       # server memory requested (per node)
#SBATCH -t 960:00:00                       # Time limit (hh:mm:ss)
#SBATCH --partition=anonymous              # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                       # Type/number of GPUs needed
#SBATCH --open-mode=append                 # Do not overwrite logs
#SBATCH --requeue                          # Requeue upon preemption

checkpoint_path="${1:-CKPT_PATH}"
seed="${2:-42}"

if [ -z "$checkpoint_path" ]; then
    echo "Usage: $0 <checkpoint_path> [seed]"
    exit 1
fi

export HYDRA_FULL_ERROR=1

# Array to store results
declare -a steps_array
declare -a ppl_array
declare -a entropy_array

# Steps to sweep: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
steps=(1 2 4 8 16 32 64 128 256 512 1024)

echo "=========================================="
echo "Starting step sweep evaluation"
echo "Checkpoint: $checkpoint_path"
echo "Seed: $seed"
echo "Steps to evaluate: ${steps[@]}"
echo "=========================================="
echo ""

# Loop through each step value
for num_steps in "${steps[@]}"; do
    echo "----------------------------------------"
    echo "Evaluating with steps=$num_steps"
    echo "----------------------------------------"
    
    # Build the command
    cmd="python -u -m main \
      mode=sample_eval \
      seed=$seed \
      model=small \
      algo=flm_shortcut \
      algo.use_curriculum=True \
      eval.checkpoint_path=$checkpoint_path \
      loader.batch_size=2 \
      loader.eval_batch_size=16 \
      sampling.num_sample_batches=8 \
      sampling.steps=$num_steps \
      sampling.noise_removal=shortcut \
      algo.flow_warmup=False \
      eval.disable_ema=False \
      algo.double_temb=False \
      algo.shortcut_on_alpha_t=True \
      algo.scale_input=False \
      +wandb.offline=true"
    
    # Print the command
    echo "Executing: $cmd"
    echo ""
    
    # Run the command and capture output
    output=$(eval "$cmd" 2>&1)
    
    # Extract Generative Perplexity (handles floating point numbers)
    ppl=$(echo "$output" | grep -i "Generative perplexity" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [ -z "$ppl" ]; then
        ppl="N/A"
    fi
    
    # Extract Sample entropy (handles floating point numbers)
    entropy=$(echo "$output" | grep -i "Sample entropy" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [ -z "$entropy" ]; then
        entropy="N/A"
    fi
    
    # Store results
    steps_array+=($num_steps)
    ppl_array+=("$ppl")
    entropy_array+=("$entropy")
    
    echo "Steps: $num_steps | Generative Perplexity: $ppl | Sample Entropy: $entropy"
    echo ""
done

# Print summary table
echo "=========================================="
echo "SUMMARY RESULTS"
echo "=========================================="
printf "%-10s %-25s %-20s\n" "Steps" "Generative Perplexity" "Sample Entropy"
echo "----------------------------------------"
for i in "${!steps_array[@]}"; do
    printf "%-10s %-25s %-20s\n" "${steps_array[$i]}" "${ppl_array[$i]}" "${entropy_array[$i]}"
done
echo "=========================================="

