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
        --add_noise=*)
            ADD_NOISE="${1#*=}" 
            ;;
        --base_dir=*)
            BASE_DIR=${1#*=} 
            ;;
        --run_name=*)
            RUN_NAME="${1#*=}" 
            ;;
        --renderer_name=*)
            RENDERER_NAME="${1#*=}" 
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
    shift
done


uv run experiments/bird/start_bird.py \
    data_path=$BASE_DIR/data/bird \
    log_path=$BASE_DIR/tinker/$RUN_NAME \
    db_path=$BASE_DIR/databases
    model_name=$MODEL_NAME \
    renderer_name=$RENDERER_NAME \
    batch_size=64 \
    group_size=8 \
    learning_rate=5e-5 \
    lora_rank=128 \
    max_tokens=3072 \
    use_kl=False \
    kl_penalty_coef=0 \
    kl_discount_factor=0 \
    eval_every=10 \
    save_every=50 \
    add_noise=$ADD_NOISE \
    wandb_name=$RUN_NAME \
    n_epochs=16 \
    timeout=60
    