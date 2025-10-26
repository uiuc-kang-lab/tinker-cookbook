set -x

# default parameters

MODEL_NAME=Qwen/Qwen3-8B
ADD_NOISE=False
BASE_DIR=/data
RUN_NAME=debug
RENDERER_NAME=qwen3
LEARNING_RATE=5e-5
NUM_DATA=-1
N_EPOCHS=16

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
        --learning_rate=*)
            LEARNING_RATE="${1#*=}" 
            ;;
        --num_data=*)
            NUM_DATA="${1#*=}" 
            ;;
        --n_epochs=*)
            N_EPOCHS="${1#*=}"
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
    db_path=$BASE_DIR/data/bird/databases \
    model_name=$MODEL_NAME \
    batch_size=64 \
    group_size=8 \
    learning_rate=$LEARNING_RATE \
    max_tokens=3072 \
    eval_every=10 \
    save_every=10 \
    add_noise=$ADD_NOISE \
    wandb_name=$RUN_NAME \
    n_epochs=$N_EPOCHS \
    timeout=60 \
    db_modification_script_path=/data/yuxuan_zhu/noisy-rl/BIRD-Platinum/db_modification \
    num_data=$NUM_DATA
    