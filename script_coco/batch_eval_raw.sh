set -x

# export CUDA_VISIBLE_DEVICES=4,5,6,7

NUM_GPUS=8
CHECKPOINT=${1}
DATA_NAME=${2}
EVALNAME=${3}
IMG_PATH=${4}
MODEL_NAME=${5}
EVAL_TYPE=${6}
BATCH_SIZE=${7:-64}
# batch size 24 for r1
CHECKPOINT="$(pwd)/${CHECKPOINT}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"


BASE_DIR="./playground/coco/$EVALNAME"

python eval/eval_${EVAL_TYPE}.py \
    --checkpoint $CHECKPOINT \
    --question-file ${DATA_NAME} \
    --answers-file ${BASE_DIR}.json \
    --image-folder ${IMG_PATH} \
    --batch-size $BATCH_SIZE \
    --model-name $MODEL_NAME \
    --num-gpus $NUM_GPUS

echo "Inference scripts have completed."