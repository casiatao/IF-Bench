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


def infer(model_name, model_path, epoch, edit_prompt_idx, src_dir, save_dir, test_json, inference_steps, height, width):
    local_model_path = None
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn, local_model_path=local_model_path),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn, local_model_path=local_model_path),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn, local_model_path=local_model_path),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/", offload_device="cpu", offload_dtype=torch.float8_e4m3fn, local_model_path=local_model_path),
    )

    # pipe.enable_vram_management(vram_limit=torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3) - 0.5)
    pipe.enable_vram_management(vram_limit=20)


    if model_name != "origin_Qwen-Image-Edit-2509" and model_path is not None:
        pipe.load_lora(pipe.dit, model_path)

    
    prompt = edit_prompt_dict[str(edit_prompt_idx)]

    test_set = json.load(open(test_json))

    for data in tqdm(test_set, desc="Infer", total=len(test_set), position=0, leave=False):
        img_path = os.path.join(src_dir, data["edit_image"])
        edit_image = Image.open(img_path).convert("RGB")
        origin_w, origin_h = edit_image.size
        edit_images = [edit_image.resize((width, height))]
        save_path = os.path.join(save_dir, f"{data['edit_image'].split('/')[-1].split('.')[0]}.jpg")
        if os.path.exists(save_path):
            continue
        image = pipe(prompt, edit_image=edit_images, seed=123, num_inference_steps=inference_steps, height=height, width=width)
        image = image.resize((origin_w, origin_h))
        image.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen-Image-Edit-2509_lora")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--edit_prompt_idx", type=int, default=1)
    parser.add_argument("--test_json", type=str, default="examples/qwen_image/model_training/validate_lora/test.json")
    parser.add_argument("--src_dir", type=str, default=None)
    parser.add_argument("--inference_step", type=int, default=40)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--save_path", type=str, default="./output/infer_results/")
    args = parser.parse_args()

    model_name = args.model_name
    model_path = args.model_path
    epoch = args.epoch
    edit_prompt_idx = args.edit_prompt_idx
    test_json = args.test_json
    src_dir = args.src_dir
    inference_steps = args.inference_step
    height = args.height
    width = args.width
    save_path = args.save_path
        
    save_dir = f"{save_path}/{model_name}/epoch{epoch}_step{inference_steps}_height{height}_width{width}/"
    os.makedirs(save_dir, exist_ok=True)

    infer(model_name, model_path, epoch, edit_prompt_idx, src_dir, save_dir, test_json, inference_steps, height, width)

