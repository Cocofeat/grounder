#!/bin/bash
set -x

# Stage-1 (single-modality) evaluation with the GroundLiver checkpoint.
# Covers all internal + external splits in a single run.
#
# Checkpoint produced by: script_coco/2025_12_14_single.sh
# Dispatcher:             script_coco/batch_eval_raw.sh
# Inference driver:       eval/eval_batch.py
# Outputs:                playground/coco/<EVAL_NAME>.json

CHECKPOINT_PATH="work_dirs/internvl_chat_v2_5/groundliver_single"

MODEL_NAME="sam"
EVAL_TYPE="batch"
BATCH_SIZE=96

# ============================================================
# Internal (labels_2025_11_29/test)
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
# External1 — HCC/ICC/small_HCC from GXMU (labels_2025_11_29/external)
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
# External batch2 — Nanning, Beihai, Guigang, Guilin, HeNan
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
# Prospective GXMU hcc_icc_mask_64 — 124 patients (49 HCC + 15 ICC + 44 Benign + 16 Normal)
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

# Pred-bbox variants (HCC/ICC use predicted bbox from the grounding model, Normal/Benign use [0,0,0,0]).
# Requires the grounding step above to have been summarized into test_*_with_pred_bbox.json first.
DATA_NAME="${DATASET_DIR}/test_QA_EN_with_pred_bbox.json"
EVAL_NAME="hcc_icc_mask_64_qa_with_pred_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATASET_DIR}/test_report_generation_EN_with_pred_bbox.json"
EVAL_NAME="hcc_icc_mask_64_report_with_pred_bbox"
script_coco/batch_eval_raw.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE
