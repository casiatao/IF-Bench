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
from threading import Thread, Lock
from multiprocessing import Process, Manager
import math

progress_bar = None
progress_lock = Lock()


def infer(model_name, model_path, epoch, edit_prompt_idx, src_dir, save_dir, test_subset, inference_steps, height, width, device_id, progress_queue):
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    print(f"[GPU {device_id}] Starting inference on {len(test_subset)} samples...")

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )

    if model_name != "origin_Qwen-Image-Edit-2509" and model_path is not None:
        pipe.load_lora(pipe.dit, model_path)

    prompt = edit_prompt_dict[str(edit_prompt_idx)]

    for data in test_subset:
        img_path = os.path.join(src_dir, data["edit_image"])
        edit_image = Image.open(img_path).convert("RGB")
        origin_w, origin_h = edit_image.size
        edit_images = [edit_image.resize((width, height))]
        save_path = os.path.join(save_dir, f"{data['edit_image'].split('/')[-1].split('.')[0]}.jpg")
        if os.path.exists(save_path):
            with progress_lock:
                progress_queue.put(1)
            continue
        image = pipe(prompt, edit_image=edit_images, seed=123,
                     num_inference_steps=inference_steps, height=height, width=width,
                     progress_bar_cmd=lambda x: tqdm(x, disable=True))
        image = image.resize((origin_w, origin_h))
        image.save(save_path)

        progress_queue.put(1)
    print(f"[GPU {device_id}] Done.")


def split_dataset(dataset, num_parts):
    chunk_size = math.ceil(len(dataset) / num_parts)
    return [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
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

    test_set = json.load(open(test_json))
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Splitting {len(test_set)} samples accordingly...")

    subsets = split_dataset(test_set, num_gpus)

    manager = Manager()
    progress_queue = manager.Queue()

    processes = []
    for device_id, subset in enumerate(subsets):
        p = Process(target=infer, args=(
            model_name, model_path, epoch, edit_prompt_idx, src_dir, save_dir,
            subset, inference_steps, height, width,
            device_id, progress_queue
        ))
        p.start()
        processes.append(p)

    from tqdm import tqdm
    with tqdm(total=len(test_set), desc="Overall Progress", ncols=100) as pbar:
        completed = 0
        while completed < len(test_set):
            progress_queue.get() 
            completed += 1
            pbar.update(1)

    for p in processes:
        p.join()

    print("✅ All GPUs finished inference.")
