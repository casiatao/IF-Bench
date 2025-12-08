import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.trainers.edit_prompts import edit_prompt_dict
from diffsynth.vram_management.layers import enable_vram_management, AutoWrappedLinear, AutoWrappedModule
from diffsynth.models.qwen_image_dit import RMSNorm
import json
import os
import argparse
from tqdm import tqdm


def infer(model_name, epoch, edit_prompt_idx, save_dir, test_json, inference_steps, height, width):
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )

    if model_name != "origin_Qwen-Image-Edit-2509":
        pipe.load_lora(pipe.dit, f"output/train_results/{model_name}/epoch-{epoch-1}.safetensors")

    prompt = edit_prompt_dict[str(edit_prompt_idx)]

    test_set = json.load(open(test_json))

    for data in tqdm(test_set, desc="Infer", total=len(test_set), position=0, leave=False):
        edit_image = Image.open(data["edit_image"]).convert("RGB")
        origin_w, origin_h = edit_image.size
        edit_images = [
            edit_image.resize((width, height)),
        ]
        save_path = os.path.join(save_dir, f"{data['edit_image'].split('/')[-1].split('.')[0]}.jpg")
        if os.path.exists(save_path):
            continue
        image = pipe(prompt, edit_image=edit_images, seed=123, num_inference_steps=inference_steps, height=height, width=width)
        image = image.resize((origin_w, origin_h))
        image.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen-Image-Edit-2509_lora")
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--edit_prompt_idx", type=int, default=1)
    parser.add_argument("--test_json", type=str, default="examples/qwen_image/model_inference/test.json")
    parser.add_argument("--inference_step", type=int, default=40)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()

    model_name = args.model_name
    epoch = args.epoch
    edit_prompt_idx = args.edit_prompt_idx
    test_json = args.test_json
        
    save_dir = f"./output/infer_results/{model_name}/epoch{epoch}_step{args.inference_step}_height{args.height}_width{args.width}/"
    os.makedirs(save_dir, exist_ok=True)

    infer(model_name, epoch, edit_prompt_idx, save_dir, test_json, args.inference_step, args.height, args.width)

