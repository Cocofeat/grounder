#!/bin/bash
set -x

# Zero-shot inference with InternVL3-8B (single-modal)
# Runs all single-modal tasks across all datasets.
# Combines: test_internal.sh + test_external_single.sh
#         + test_external_batch2.sh + test_hcc_icc_mask_64.sh
#
# After inference, copy results to evaluation_archive:
#   for f in playground/coco/*.json; do
#     # copy to the appropriate evaluation_archive/single/*/pred_zeroshot/ dir
#   done
#
# Then run bootstrap CI:
#   python -u tools_coco/bootstrap_ci.py --mode single --n-iterations 1000 \
#       --pred-subdir pred_zeroshot

CHECKPOINT_PATH="pretrained/InternVL3-8B"

MODEL_NAME="internvl3"
EVAL_TYPE="batch"
BATCH_SIZE=96

# ============================================================
# Internal
# ============================================================
DATA_BASE="/mnt/data/by/data/coco_new/labels_2025_11_29"
DATA_DIR="${DATA_BASE}/test"
IMG_PATH="/mnt/data/by/data/coco_new/data"

DATA_NAME="${DATA_DIR}/test_QA_EN.json"
EVAL_NAME="internal_test_qa"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_QA_EN_with_bbox.json"
EVAL_NAME="internal_test_qa_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_bbox_v2.json"
EVAL_NAME="internal_test_grounding"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN.json"
EVAL_NAME="internal_test_report"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_with_bbox.json"
EVAL_NAME="internal_test_report_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# ============================================================
# External1 (HCC/ICC/small_HCC from GXMU)
# ============================================================
DATA_DIR="${DATA_BASE}/external"
IMG_PATH="/mnt/data/by/data/coco_new/data_external_final"

DATA_NAME="${DATA_DIR}/test_QA_EN_external.json"
EVAL_NAME="external1_single_qa"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_QA_EN_external_with_bbox.json"
EVAL_NAME="external1_single_qa_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_bbox.json"
EVAL_NAME="external1_single_grounding"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN.json"
EVAL_NAME="external1_single_report"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_with_bbox.json"
EVAL_NAME="external1_single_report_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# ============================================================
# External ENSHI
# ============================================================
DATA_DIR="${DATA_BASE}/external_ENSHI"
IMG_PATH="/mnt/data/by/data/coco_new/external"

DATA_NAME="${DATA_DIR}/test_QA_EN.json"
EVAL_NAME="external_ENSHI_single_qa"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_QA_EN_with_bbox.json"
EVAL_NAME="external_ENSHI_single_qa_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_bbox.json"
EVAL_NAME="external_ENSHI_single_grounding"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN.json"
EVAL_NAME="external_ENSHI_single_report"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_with_bbox.json"
EVAL_NAME="external_ENSHI_single_report_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# ============================================================
# External GXMU benign/normal
# ============================================================
DATA_DIR="${DATA_BASE}/external_GXMU_benign_normal"
IMG_PATH="/mnt/data/by/data/coco_new/external"

DATA_NAME="${DATA_DIR}/test_QA_EN.json"
EVAL_NAME="external_GXMU_bn_single_qa"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_QA_EN_with_bbox.json"
EVAL_NAME="external_GXMU_bn_single_qa_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_bbox.json"
EVAL_NAME="external_GXMU_bn_single_grounding"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN.json"
EVAL_NAME="external_GXMU_bn_single_report"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_with_bbox.json"
EVAL_NAME="external_GXMU_bn_single_report_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# ============================================================
# External SanYa
# ============================================================
DATA_DIR="${DATA_BASE}/external_SanYa"
IMG_PATH="/mnt/data/by/data/coco_new/external"

DATA_NAME="${DATA_DIR}/test_QA_EN.json"
EVAL_NAME="external_SanYa_single_qa"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_QA_EN_with_bbox.json"
EVAL_NAME="external_SanYa_single_qa_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_bbox.json"
EVAL_NAME="external_SanYa_single_grounding"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN.json"
EVAL_NAME="external_SanYa_single_report"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_with_bbox.json"
EVAL_NAME="external_SanYa_single_report_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# ============================================================
# External batch2 (Nanning, Beihai, Guigang, Guilin, HeNan)
# ============================================================
DATA_BASE="/mnt/data/by/data/coco_new/labels_2026_1_17"
IMG_PATH="/mnt/data/by/data/coco_new/data_external_batch2"

for DATASET_DIR in "${DATA_BASE}"/liver_MRI_EXTERNAL*_ready; do
  if [ ! -d "${DATASET_DIR}" ]; then
    continue
  fi

  DATA_NAME="${DATASET_DIR}/test_QA_EN.json"
  EVAL_NAME="external_batch2_$(basename "${DATASET_DIR}")_qa"
  script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

  DATA_NAME="${DATASET_DIR}/test_QA_EN_with_bbox.json"
  EVAL_NAME="external_batch2_$(basename "${DATASET_DIR}")_qa_with_bbox"
  script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

  DATA_NAME="${DATASET_DIR}/test_report_generation_EN.json"
  EVAL_NAME="external_batch2_$(basename "${DATASET_DIR}")_report"
  script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

  DATA_NAME="${DATASET_DIR}/test_report_generation_EN_with_bbox.json"
  EVAL_NAME="external_batch2_$(basename "${DATASET_DIR}")_report_with_bbox"
  script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE
done

# ============================================================
# Prospective GXMU (hcc_icc_mask_64)
# 124 patients: 49 HCC + 15 ICC + 44 Benign + 16 Normal
# ============================================================
DATASET_DIR="${DATA_BASE}/liver_MRI_EXTERNAL_prospective_GXMU_ready_hcc_icc_mask_64"

DATA_NAME="${DATASET_DIR}/test_QA_EN.json"
EVAL_NAME="hcc_icc_mask_64_qa"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATASET_DIR}/test_QA_EN_with_bbox.json"
EVAL_NAME="hcc_icc_mask_64_qa_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATASET_DIR}/test_report_generation_EN.json"
EVAL_NAME="hcc_icc_mask_64_report"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATASET_DIR}/test_report_generation_EN_with_bbox.json"
EVAL_NAME="hcc_icc_mask_64_report_with_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATASET_DIR}/test_bbox.json"
EVAL_NAME="hcc_icc_mask_64_grounding"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# Pred bbox tasks (need predicted bbox from grounding model — uncomment if available)
# DATA_NAME="${DATASET_DIR}/test_QA_EN_with_pred_bbox.json"
# EVAL_NAME="hcc_icc_mask_64_qa_with_pred_bbox"
# script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# DATA_NAME="${DATASET_DIR}/test_report_generation_EN_with_pred_bbox.json"
# EVAL_NAME="hcc_icc_mask_64_report_with_pred_bbox"
# script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE
