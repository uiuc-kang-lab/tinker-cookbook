set -x

# default parameters

MODEL_NAME=Qwen/Qwen3-8B
ADD_NOISE=False
BASE_DIR=/data
RUN_NAME=debug

while [[ "$1" == --* ]]; do
    case "$1" in
        --model=*)
            MODEL_NAME="${1#*=}" 
            ;;
        --add_noise=*)
            ADD_NOISE="${1#*=}" 
            ;;
        --base_dir=*)
            BASE_DIR=${1#*=} 
            ;;
        --run_name=*)
            RUN_NAME="${1#*=}" 
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
    shift
done


uv run experiments/deepscaler_math/start_deepscaler.py \
    data_path=$BASE_DIR/data/deepscaler \
    log_path=$BASE_DIR/tinker/$RUN_NAME \
    model_name=$MODEL_NAME \
    batch_size=64 \
    group_size=8 \
    learning_rate=1e-6 \
    lora_rank=32 \
    max_tokens=3072 \
    use_kl=False \
    kl_penalty_coef=0 \
    kl_discount_factor=0 \
    num_substeps=1 \
    remove_constant_reward_groups=False \
    eval_interval=10 \
    save_interval=50 \
    add_noise=$ADD_NOISE \
    wandb_name=$RUN_NAME 
    