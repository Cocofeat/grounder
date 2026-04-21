import argparse
import os
import json
from typing import List, Dict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from internvl.model import load_model_and_tokenizer_gating as load_model_and_tokenizer

from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode


class RGBConverter:
    def __call__(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def build_picklable_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = T.Compose([
        RGBConverter(),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


class MultiImageQuestionDataset(Dataset):
    """Dataset for multi-image QA, where each question has multiple images."""

    def __init__(self, questions: List[Dict], image_folder: str, image_size: int):
        self.questions = questions
        self.image_folder = image_folder
        self.transform = build_picklable_transform(input_size=image_size)
        self.image_size = image_size

    def __len__(self):
        return len(self.questions)

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.transform(image)
        return pixel_values

    def __getitem__(self, idx):
        question = self.questions[idx]

        # Handle both single image (string) and multiple images (list)
        image_paths = question['image']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Load all images for this question
        pixel_values_list = []
        for img_path in image_paths:
            full_path = os.path.join(self.image_folder, img_path)
            pv = self.load_image(full_path)
            pixel_values_list.append(pv)

        # Stack all images: [num_images, C, H, W]
        pixel_values = torch.stack(pixel_values_list, dim=0)

        return {
            'pixel_values': pixel_values,
            'num_images': len(image_paths),
            'question_id': question['question_id'],
            'prompt': question['text'],
            'label': question.get('label', None),
            'original_idx': idx
        }


def collate_fn(batch):
    """Custom collate function to handle variable number of images per question."""
    # Concatenate all pixel values
    all_pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)

    # Record number of images per sample
    num_images_list = [item['num_images'] for item in batch]

    return {
        'pixel_values': all_pixel_values,
        'num_images_list': num_images_list,
        'question_id': [item['question_id'] for item in batch],
        'prompt': [item['prompt'] for item in batch],
        'label': [item['label'] for item in batch],
        'original_idx': [item['original_idx'] for item in batch]
    }


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_inference(rank, world_size, args):
    setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    model, tokenizer = load_model_and_tokenizer(args)
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    image_size = model.module.config.force_image_size or model.module.config.vision_config.image_size \
        if isinstance(model, DDP) else model.config.force_image_size or model.config.vision_config.image_size

    with open(args.question_file) as f:
        questions = json.load(f)

    dataset = MultiImageQuestionDataset(questions, args.image_folder, image_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    results = []

    model.eval()
    if rank == 0:
        progress_bar = tqdm(total=len(dataloader), desc=f'Rank {rank} Processing')

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
            prompts = batch['prompt']
            num_images_list = batch['num_images_list']

            generation_config = dict(
                do_sample=True if args.temperature > 0 else False,
                num_beams=args.num_beams,
                max_new_tokens=256
            )

            # For multi-image: num_patches_list should indicate number of images per question
            # Each image is treated as 1 patch in this case
            num_patches_list = num_images_list

            # Batch processing
            responses = model.module.batch_chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                questions=prompts,
                num_patches_list=num_patches_list,
                generation_config=generation_config,
                verbose=False
            ) if isinstance(model, DDP) else model.batch_chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                questions=prompts,
                num_patches_list=num_patches_list,
                generation_config=generation_config,
                verbose=False
            )

            # Store results with original indices
            for i in range(len(responses)):
                result = {
                    "question_id": batch['question_id'][i],
                    "prompt": batch['prompt'][i],
                    "text": responses[i],
                    "metadata": {},
                    "original_idx": batch['original_idx'][i]
                }
                results.append(result)

            if rank == 0:
                progress_bar.update(1)

    if rank == 0:
        progress_bar.close()

    # Save results from this rank
    output_file = f"{args.answers_file}_rank{rank}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f)

    cleanup()


def merge_results(args, world_size, questions):
    """Merge results from all ranks in the correct order"""
    all_results = []
    for rank in range(world_size):
        rank_file = f"{args.answers_file}_rank{rank}.json"
        with open(rank_file) as f:
            results = json.load(f)
            all_results.extend(results)
        os.remove(rank_file)

    all_results.sort(key=lambda x: x["original_idx"])

    seen = {}
    unique_results = []
    for item in all_results:
        if item["original_idx"] not in seen:
            seen[item["original_idx"]] = True
            unique_results.append(item)
    all_results = unique_results

    print(f"Total results: {len(all_results)}")
    print(f"Total questions: {len(questions)}")

    # Verify order matches original questions
    for i, (result, question) in enumerate(zip(all_results, questions)):
        assert str(result["question_id"]) == str(question["question_id"]), \
            f"Result mismatch at position {i}: {result['question_id']} != {question['question_id']}"

    # Remove original_idx before saving
    for result in all_results:
        del result["original_idx"]

    # Write final results
    with open(args.answers_file, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--answers-file', type=str, default='./Your_Results')
    parser.add_argument('--question-file', type=str, default='./Your_Results')
    parser.add_argument('--image-folder', type=str, default='/path/to/images')
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--model-name', type=str, default='chat')
    parser.add_argument('--task-type', type=str, default='cls')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    with open(args.question_file) as f:
        questions = json.load(f)

    mp.spawn(
        run_inference,
        args=(args.num_gpus, args),
        nprocs=args.num_gpus,
        join=True
    )

    merge_results(args, args.num_gpus, questions)
    print('Finished multi-image inference')
