#!/bin/bash
set -x

# Stage-2b (multi-modality gating) evaluation with the GroundLiver gating checkpoint.
# Covers all internal + external multimodal splits in a single run.
#
# Checkpoint produced by: script_coco/2025_12_14_multi_gating.sh
# Dispatcher:             script_coco/batch_eval_multi.sh
# Inference driver:       eval/eval_multi_batch.py
#   NOTE: when evaluating a gating checkpoint you likely need to swap the
#         import on line 10 to `load_model_and_tokenizer_gating`.
# Outputs:                playground/coco/<EVAL_NAME>.json

CHECKPOINT_PATH="work_dirs/internvl_chat_v2_5/internvl3_coco_2025_12_14_multi_gating"

MODEL_NAME="sam"
EVAL_TYPE="multi_batch"
BATCH_SIZE=24

# ============================================================
# Internal multimodal (labels_2025_11_29/multimodal_test)
# ============================================================
DATA_BASE="/mnt/data/by/data/coco_new/labels_2025_11_29"
DATA_DIR="${DATA_BASE}/multimodal_test"
IMG_PATH="/mnt/data/by/data/coco_new/data"

DATA_NAME="${DATA_DIR}/test_QA_EN_multimodal_4mod.json"
EVAL_NAME="internal_multi_qa_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_QA_EN_multimodal_4mod_with_bbox.json"
EVAL_NAME="internal_multi_qa_with_bbox_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_grounding_EN_multimodal_4mod_QA.json"
EVAL_NAME="internal_multi_grounding_qa_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_grounding_EN_multimodal_4mod_report.json"
EVAL_NAME="internal_multi_grounding_report_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_multimodal.json"
EVAL_NAME="internal_multi_report_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_multimodal_with_bbox.json"
EVAL_NAME="internal_multi_report_with_bbox_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# ============================================================
# External1 — HCC/ICC/small_HCC from GXMU (labels_2025_11_29/external_multimodal)
# ============================================================
DATA_DIR="${DATA_BASE}/external_multimodal"
IMG_PATH="/mnt/data/by/data/coco_new/data_external_final"

DATA_NAME="${DATA_DIR}/test_QA_EN_external_multimodal_4mod.json"
EVAL_NAME="external1_multi_qa_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_QA_EN_external_multimodal_4mod_with_bbox.json"
EVAL_NAME="external1_multi_qa_with_bbox_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_grounding_multimodal_4mod.json"
EVAL_NAME="external1_multi_grounding_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_multimodal.json"
EVAL_NAME="external1_multi_report_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_multimodal_with_bbox.json"
EVAL_NAME="external1_multi_report_with_bbox_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# ============================================================
# External ENSHI multimodal
# ============================================================
DATA_DIR="${DATA_BASE}/external_ENSHI_multimodal"
IMG_PATH="/mnt/data/by/data/coco_new/external"

DATA_NAME="${DATA_DIR}/test_QA_EN_multimodal_4mod.json"
EVAL_NAME="external_ENSHI_multi_qa_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_QA_EN_multimodal_4mod_with_bbox.json"
EVAL_NAME="external_ENSHI_multi_qa_with_bbox_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_grounding_multimodal_4mod.json"
EVAL_NAME="external_ENSHI_multi_grounding_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_multimodal.json"
EVAL_NAME="external_ENSHI_multi_report_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_multimodal_with_bbox.json"
EVAL_NAME="external_ENSHI_multi_report_with_bbox_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# ============================================================
# External GXMU benign/normal multimodal
# ============================================================
DATA_DIR="${DATA_BASE}/external_GXMU_benign_normal_multimodal"
IMG_PATH="/mnt/data/by/data/coco_new/external"

DATA_NAME="${DATA_DIR}/test_QA_EN_multimodal_4mod.json"
EVAL_NAME="external_GXMU_bn_multi_qa_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_QA_EN_multimodal_4mod_with_bbox.json"
EVAL_NAME="external_GXMU_bn_multi_qa_with_bbox_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_grounding_multimodal_4mod.json"
EVAL_NAME="external_GXMU_bn_multi_grounding_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_multimodal.json"
EVAL_NAME="external_GXMU_bn_multi_report_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_multimodal_with_bbox.json"
EVAL_NAME="external_GXMU_bn_multi_report_with_bbox_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# ============================================================
# External SanYa multimodal
# ============================================================
DATA_DIR="${DATA_BASE}/external_SanYa_multimodal"
IMG_PATH="/mnt/data/by/data/coco_new/external"

DATA_NAME="${DATA_DIR}/test_QA_EN_multimodal_4mod.json"
EVAL_NAME="external_SanYa_multi_qa_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_QA_EN_multimodal_4mod_with_bbox.json"
EVAL_NAME="external_SanYa_multi_qa_with_bbox_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_grounding_multimodal_4mod.json"
EVAL_NAME="external_SanYa_multi_grounding_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_multimodal.json"
EVAL_NAME="external_SanYa_multi_report_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATA_DIR}/test_report_generation_EN_multimodal_with_bbox.json"
EVAL_NAME="external_SanYa_multi_report_with_bbox_gating"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# ============================================================
# External batch2 multimodal (Nanning, Beihai, Guigang, Guilin, HeNan)
# ============================================================
DATA_BASE="/mnt/data/by/data/coco_new/labels_2026_1_17"
IMG_PATH="/mnt/data/by/data/coco_new/data_external_batch2"

for DATASET_DIR in "${DATA_BASE}"/liver_MRI_EXTERNAL*_ready_multimodal; do
  if [ ! -d "${DATASET_DIR}" ]; then
    continue
  fi

  DATA_NAME="${DATASET_DIR}/test_QA_EN_multimodal_4mod.json"
  EVAL_NAME="external_batch2_multi_$(basename "${DATASET_DIR}")_qa"
  script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

  DATA_NAME="${DATASET_DIR}/test_QA_EN_multimodal_4mod_with_bbox.json"
  EVAL_NAME="external_batch2_multi_$(basename "${DATASET_DIR}")_qa_with_bbox"
  script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

  DATA_NAME="${DATASET_DIR}/test_report_generation_EN_multimodal.json"
  EVAL_NAME="external_batch2_multi_$(basename "${DATASET_DIR}")_report"
  script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

  DATA_NAME="${DATASET_DIR}/test_report_generation_EN_multimodal_with_bbox.json"
  EVAL_NAME="external_batch2_multi_$(basename "${DATASET_DIR}")_report_with_bbox"
  script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE
done

# ============================================================
# Prospective GXMU hcc_icc_mask_64 multimodal — 124 patients
# ============================================================
DATASET_DIR="${DATA_BASE}/liver_MRI_EXTERNAL_prospective_GXMU_ready_hcc_icc_mask_64_multimodal"

DATA_NAME="${DATASET_DIR}/test_QA_EN_multimodal_4mod.json"
EVAL_NAME="hcc_icc_mask_64_multi_qa"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATASET_DIR}/test_QA_EN_multimodal_4mod_with_bbox.json"
EVAL_NAME="hcc_icc_mask_64_multi_qa_with_bbox"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATASET_DIR}/test_report_generation_EN_multimodal.json"
EVAL_NAME="hcc_icc_mask_64_multi_report"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATASET_DIR}/test_report_generation_EN_multimodal_with_bbox.json"
EVAL_NAME="hcc_icc_mask_64_multi_report_with_bbox"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATASET_DIR}/test_grounding_multimodal_4mod.json"
EVAL_NAME="hcc_icc_mask_64_multi_grounding"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

# Pred-bbox variants (HCC/ICC use predicted bbox, Normal/Benign use [0,0,0,0]).
DATA_NAME="${DATASET_DIR}/test_QA_EN_multimodal_4mod_with_pred_bbox.json"
EVAL_NAME="hcc_icc_mask_64_multi_qa_with_pred_bbox"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE

DATA_NAME="${DATASET_DIR}/test_report_generation_EN_multimodal_with_pred_bbox.json"
EVAL_NAME="hcc_icc_mask_64_multi_report_with_pred_bbox"
script_coco/batch_eval_multi.sh $CHECKPOINT_PATH $DATA_NAME $EVAL_NAME $IMG_PATH $MODEL_NAME $EVAL_TYPE $BATCH_SIZE
