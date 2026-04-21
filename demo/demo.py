"""Single-process demo for reviewing LiverGrounder inference.

Runs one or all of the 4 bundled samples (one per class: HCC / ICC / Benign /
Normal) through either the single-modality checkpoint or the multi-modality
gating checkpoint and prints the prompt, ground-truth label, and model answer
side by side.

Usage:
    python demo/demo.py --stage single                  # all 4 single-mod samples
    python demo/demo.py --stage multi                   # all 4 multi-mod samples (with bbox)
    python demo/demo.py --stage single --sample-idx 2   # just one sample
    python demo/demo.py --stage multi  --checkpoint work_dirs/other

Requires one visible GPU (uses cuda:0). No torchrun / DDP.
"""
import argparse
import json
import os
import sys
from typing import List

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda im: im.convert('RGB') if im.mode != 'RGB' else im),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_stack(image_paths: List[str], image_folder: str, input_size: int) -> torch.Tensor:
    transform = build_transform(input_size)
    tensors = []
    for rel in image_paths:
        im = Image.open(os.path.join(image_folder, rel))
        tensors.append(transform(im))
    return torch.stack(tensors, dim=0)


def load_single(checkpoint: str):
    from internvl.model.internvl_chat import InternVLChatModel
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
    ).eval().cuda()
    return model, tokenizer


def load_gating(checkpoint: str):
    from internvl.model.internvl_chat.modeling_internvl_chat_gating import InternVLChatModelGating
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModelGating.from_pretrained(
        checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
    ).eval().cuda()
    return model, tokenizer


def run(args):
    if args.stage == 'single':
        samples_path = os.path.join(HERE, 'samples_single.json')
        default_ckpt = 'work_dirs/internvl_chat_v2_5/groundliver_single'
        model, tokenizer = load_single(args.checkpoint or default_ckpt)
    else:
        samples_path = os.path.join(HERE, 'samples_multi.json')
        default_ckpt = 'work_dirs/internvl_chat_v2_5/internvl3_coco_2025_12_14_multi_gating'
        model, tokenizer = load_gating(args.checkpoint or default_ckpt)

    with open(samples_path) as f:
        samples = json.load(f)

    if args.sample_idx is not None:
        samples = [samples[args.sample_idx]]

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    image_folder = args.image_folder or os.path.join(HERE, 'images')

    generation_config = dict(
        do_sample=False,
        num_beams=1,
        max_new_tokens=256,
    )

    for s in samples:
        image_paths = s['image'] if isinstance(s['image'], list) else [s['image']]
        pixel_values = load_image_stack(image_paths, image_folder, image_size)
        pixel_values = pixel_values.to('cuda', dtype=torch.bfloat16)

        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=s['text'],
            generation_config=generation_config,
            num_patches_list=[len(image_paths)],
            verbose=False,
        )

        print('=' * 80)
        print(f"question_id : {s['question_id']}")
        print(f"label class : {s.get('patient_label')}")
        print(f"modalities  : {s.get('modalities', [s.get('modality')])}")
        print(f"images      : {image_paths}")
        print('-' * 80)
        print('prompt:')
        print(s['text'])
        print('-' * 80)
        print(f"ground truth : {s.get('label')}")
        print(f"model answer : {response}")
    print('=' * 80)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['single', 'multi'], required=True,
                   help='single = stage-1 single-modality checkpoint; multi = stage-2b gating checkpoint')
    p.add_argument('--checkpoint', default=None,
                   help='Override the default checkpoint path for the selected stage.')
    p.add_argument('--image-folder', default=None,
                   help='Override demo/images (paths inside samples_*.json are resolved relative to this).')
    p.add_argument('--sample-idx', type=int, default=None,
                   help='Run just one sample by index (0-3). Omit to run all 4.')
    run(p.parse_args())


if __name__ == '__main__':
    main()
