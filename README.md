# LiverGrounder

Code release for **LiverGrounder**, a liver lesion multi-modal vision-language model.

- **Stage 1** — single-modality fine-tune across CT phases (PRE / AP / PVP) and MRI T2WI.
- **Stage 2** — multi-modality inference with a lightweight per-modality gate that re-weights each phase's contribution before the language model.

This repository contains the Python package, inference drivers, evaluation pipeline, and a small self-contained **demo** with four real liver patient cases (one per class: HCC / ICC / Benign / Normal).

## Setup

```bash
git clone <this-repo>.git LiverGrounder
cd LiverGrounder

# Install Python dependencies (torch, transformers==4.37.2, deepspeed, peft, timm, einops, …).
pip install -e .
pip install flash-attn --no-build-isolation

# Download the InternVL3-8B base weights.
mkdir -p pretrained && cd pretrained
huggingface-cli download --resume-download --local-dir-use-symlinks False \
    OpenGVLab/InternVL3-8B --local-dir InternVL3-8B
cd ..
```

Hardware: one NVIDIA GPU with ≥16 GB of VRAM is enough for demo / inference (bf16).

## Demo (4 bundled patients)

The [demo/](demo/) folder ships 16 PNG slices (four modalities × four patients) plus matched JSON prompts, so reviewers can run inference without any additional data.

```bash
# Single-modality (stage 1) — one AP image per patient
python demo/demo.py --stage single

# Multi-modality gating (stage 2) — all four phases per patient
python demo/demo.py --stage multi

# Just one sample
python demo/demo.py --stage multi --sample-idx 2

# Zero-shot baseline against the un-tuned InternVL3-8B
python demo/demo.py --stage single --checkpoint pretrained/InternVL3-8B
```

Each call prints the prompt, ground-truth label, and model answer. In `--stage multi` the script also prints the per-modality gating weights, e.g. `gating values (per image): [0.92, 0.91, 0.92, 0.93]`.

Default checkpoint paths are `work_dirs/internvl_chat_v2_5/groundliver_single` (stage 1) and `work_dirs/internvl_chat_v2_5/internvl3_coco_2025_12_14_multi_gating` (stage 2). Use `--checkpoint` to point elsewhere. See [demo/README.md](demo/README.md) for the full field-by-field description of the samples.

## Test suite

Evaluation runs in three phases. All commands are launched from the repo root.

### Phase 1 — inference (writes answer JSONs to `playground/coco/`)

Three consolidated runners cover the full internal + external sweep:

| Runner | Stage | Driver |
|---|---|---|
| [`script_coco/test_single.sh`](script_coco/test_single.sh) | Stage 1 | [`eval/eval_batch.py`](eval/eval_batch.py) |
| [`script_coco/test_gating.sh`](script_coco/test_gating.sh) | Stage 2 | [`eval/eval_multi_batch.py`](eval/eval_multi_batch.py) |
| [`script_coco/test_zeroshot_single.sh`](script_coco/test_zeroshot_single.sh) | Zero-shot baseline | [`eval/eval_batch.py`](eval/eval_batch.py) |

Each runner iterates over internal + 4 external cohorts + a batch-2 loop + the prospective-GXMU `hcc_icc_mask_64` split, and within each cohort over the task variants QA / QA-with-bbox / grounding / report / report-with-bbox. Outputs land in `playground/coco/<EVAL_NAME>.json`.

> **Note:** the test sweeps read from a private liver CT / MRI cohort whose paths are configured in `shell/data/*.json`. Reviewers cannot re-execute them without that data, but the scripts and drivers are shipped so the pipeline shape is fully inspectable. For end-to-end functional verification, use the [demo/](demo/) instead.

### Phase 2 — per-run scoring ([tools_coco/](tools_coco/))

```bash
python tools_coco/eval_coco_qa_single_vote_f1.py  ...   # QA F1 with patient-level voting
python tools_coco/eval_report_single_vote.py       ...   # Report generation metrics
python tools_coco/eval_grounding.py                ...   # Visual grounding (IoU, recall)
python tools_coco/bootstrap_ci.py                  ...   # 95 % CI over any of the above
```

### Phase 3 — aggregation / export

```bash
python tools_coco/summarize_all_results.py                # Table across all cohorts
python tools_coco/export_all_results.py                   # Per-patient breakdown
```

## Repository layout

```
LiverGrounder/
├── internvl/         # trimmed InternVL3 package (chat + gating head)
├── eval/             # two inference drivers (eval_batch, eval_multi_batch)
├── script_coco/      # training + evaluation shell scripts
├── tools_coco/       # scoring + aggregation Python scripts
├── shell/data/       # dataset meta configs for training / testing
├── demo/             # bundled reviewer demo (images + demo.py)
├── pretrained/       # InternVL3-8B goes here (see Setup)
├── playground/coco/  # eval answer JSONs land here
├── zero_stage1_config.json
└── pyproject.toml
```

## License

See [LICENSE](LICENSE). The underlying InternVL3-8B weights follow the license of [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B).
