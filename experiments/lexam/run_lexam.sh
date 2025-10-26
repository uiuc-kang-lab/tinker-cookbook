set -x

# default parameters

MODEL_NAME=Qwen/Qwen3-8B
ADD_NOISE=False
BASE_DIR=/data
RUN_NAME=debug
RENDERER_NAME=qwen3

while [[ "$1" == --* ]]; do
    case "$1" in
        --model=*)
            MODEL_NAME="${1#*=}" 
            ;;
        --noise_rate=*)
            NOISE_RATE="${1#*=}" 
            ;;
        --base_dir=*)
            BASE_DIR=${1#*=} 
            ;;
        --run_name=*)
            RUN_NAME="${1#*=}" 
            ;;
        --group_size=*)
            GROUP_SIZE="${1#*=}" 
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
    shift
done


uv run experiments/lexam/start_lexam.py \
    data_path=$BASE_DIR/data/lexam \
    log_path=$BASE_DIR/tinker/$RUN_NAME \
    model_name=$MODEL_NAME \
    batch_size=64 \
    group_size=$GROUP_SIZE \
    learning_rate=5e-4 \
    max_tokens=3072 \
    eval_interval=10 \
    save_interval=50 \
    noise_rate=$NOISE_RATE \
    wandb_name=$RUN_NAME \
    n_epochs=20
    