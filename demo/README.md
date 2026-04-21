# LiverGrounder demo

A self-contained smoke test for reviewers. Ships with **4 randomly-selected liver MRI/CT multi-phase samples** — one per class (HCC, ICC, Benign, Normal) — drawn from the private training cohort, plus a small one-shot inference script.

```
demo/
├── demo.py                 # single-process, single-GPU inference
├── samples_multi.json      # 4 samples, 4 modalities each (PRE, AP, PVP, T2WI) with bbox prompts
├── samples_single.json     # same 4 patients, using only the AP phase (no bbox)
├── images/                 # 16 PNG slices (~0.6 MB total)
│   └── <patient_id>/<modality>/images/<slice>.png
└── README.md
```

Multi samples are drawn from the `train_QA_EN_multimodal_4mod_with_bbox.jsonl` split, so each prompt includes a `Detected lesions (if any) with bounding boxes` block that anchors the answer to a specific phase (usually AP).

## Requirements

- One GPU with ≥16 GB VRAM (bf16; the 8B base takes ~15 GB during inference)
- `torch>=2`, `transformers==4.37.2`, `timm`, `peft`, `einops`, `accelerate`, optionally `flash-attn`
- The two fine-tuned checkpoints at their default paths:
  - `work_dirs/internvl_chat_v2_5/groundliver_single` (Stage 1)
  - `work_dirs/internvl_chat_v2_5/internvl3_coco_2025_12_14_multi_gating` (Stage 2)

If you only have `pretrained/InternVL3-8B`, pass `--checkpoint pretrained/InternVL3-8B` to see the zero-shot baseline.

## Run

From the repo root:

```bash
cd /path/to/LiverGrounder

# Single-modality demo — uses the AP (arterial phase) image per patient
python demo/demo.py --stage single

# Multi-modality gating demo — uses all 4 phases jointly (with bbox prompts)
python demo/demo.py --stage multi

# Just one sample
python demo/demo.py --stage multi --sample-idx 2

# Point at a different checkpoint (e.g. the zero-shot baseline)
python demo/demo.py --stage single --checkpoint pretrained/InternVL3-8B
```

## What each sample asks

All 4 samples pose QA1 — the focal-lesion screening question:

> Does this patient have a focal liver lesion?
> A. There are a focal lesion or multipul focal lesions;
> B. No focal lesions are present

Both sample files include the ground-truth `label` and a `patient_label` key (HCC/ICC/Benign/Normal) so you can eyeball whether the prediction matches the clinical ground truth.

## Example output

```
================================================================================
question_id : demo_multi_00_202503130398
label class : HCC
modalities  : ['PRE', 'AP', 'PVP', 'T2WI']
images      : ['202503130398/PRE/images/58.png', ... 4 paths ...]
--------------------------------------------------------------------------------
prompt:
<image>
<image>
<image>
<image>
The four images above correspond to:
1) CT pre-contrast (PRE)
2) CT arterial phase (AP)
3) CT portal venous phase (PVP)
4) MRI T2-weighted (T2WI)

Detected lesions (if any) with bounding boxes:
PVP: <box>[[268, 537, 422, 670]]</box>
Question: Does this patient have a focal liver lesion? ...
--------------------------------------------------------------------------------
ground truth : A. There are a focal lesion or multipul focal lesions
model answer : A. There are a focal lesion or multipul focal lesions
================================================================================
```

For `--stage multi`, the gating model also prints a one-line `gating values (per image): [...]` before each answer — the per-modality weights learned by the modality gate (≈1.0 means the modality is fully attended to, ≈0 means suppressed).

```
gating values (per image): [0.91, 0.88, 0.88, 0.89]
```

## Notes on reproducibility

- Sample selection used `random.seed(20260421)` and picked one patient per class — the patient IDs above are deterministic.
- All 16 PNGs are drawn directly from `train_QA_EN_multimodal_4mod_with_bbox.jsonl`; the rest of the training set is not redistributed.
- Inference uses greedy decoding (`do_sample=False, num_beams=1, max_new_tokens=256`), so outputs are deterministic for a given checkpoint and hardware stack.
