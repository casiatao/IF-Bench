import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import re

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_internvl35_1b():
    path = 'OpenGVLab/InternVL3_5-1B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def load_internvl35_2b():
    path = 'OpenGVLab/InternVL3_5-2B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def load_internvl35_4b():
    path = 'OpenGVLab/InternVL3_5-4B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def load_internvl35_8b():
    path = 'OpenGVLab/InternVL3_5-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def load_internvl35_14b():
    path = 'OpenGVLab/InternVL3_5-14B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer



def load_internvl35_38b():
    path = 'OpenGVLab/InternVL3_5-38B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def load_internvl35_20b_a4b():
    path = 'OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def load_internvl35_30b_a3b():
    path = 'OpenGVLab/InternVL3_5-30B-A3B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def load_internvl35_241b_a28b():
    path = 'OpenGVLab/InternVL3_5-241B-A28B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def internvl35_infer_no_think(model, processor, system_prompt, img_path=None, max_new_tokens=1024):
    if img_path is not None:
        img_path = [img_path] if isinstance(img_path, str) else img_path
        pixel_values_list = []
        for i in range(len(img_path)):
            pixel_values = load_image(img_path[i], max_num=12).to(torch.bfloat16).cuda()
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list, dim=0)
    else:
        pixel_values = None
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True, eos_token_id=151645, pad_token_id=151645)

    response = model.chat(processor, pixel_values, system_prompt, generation_config)
    return response



def internvl35_infer_think(model, processor, system_prompt, img_path=None, max_new_tokens=1024):
    R1_SYSTEM_PROMPT = """
    You are an AI assistant that rigorously follows this response protocol:

    1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

    2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

    Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
    """.strip()

    model.system_message = R1_SYSTEM_PROMPT

    if img_path is not None:
        img_path = [img_path] if isinstance(img_path, str) else img_path
        pixel_values_list = []
        for i in range(len(img_path)):
            pixel_values = load_image(img_path[i], max_num=12).to(torch.bfloat16).cuda()
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list, dim=0)
    else:
        pixel_values = None
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, eos_token_id=151645, pad_token_id=151645)

    response = model.chat(processor, pixel_values, system_prompt, generation_config)

    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    if "\boxed" in response:
        response = re.search(r"\boxed{(.*?)}", response).group(1)
        response = response.strip()

    return response


if __name__ == "__main__":
    model, processor = load_internvl35_1b()
    # model, processor = load_internvl35_2b()
    # model, processor = load_internvl35_4b()
    # model, processor = load_internvl35_8b()
    # model, processor = load_internvl35_14b()
    # model, processor = load_internvl35_38b()
    # model, processor = load_internvl35_20b_a4b()
    # model, processor = load_internvl35_30b_a3b()
    # model, processor = load_internvl35_241b_a28b()