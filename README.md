# IF-Bench

## 📖 Introduction
This repository contains the official evaluation implementation of IF-Bench, the first high-quality benchmark for evaluating multimodal understanding of infrared images, and the training implementation of editing models in GenViP.
<p align="center">
<img src="imgs/intro.png" alt="intro" style="width:70%; height:auto;" />
</p>


## 📓 Environment Setup
```
# 1. create conda environment
conda create -n if_bench python=3.9
conda activate if_bench

# 2. install packages for if_bench evaluation
bash evaluate/set_env.sh

# 3. (optional) install packages for edit fine-tuning
cd edit_ft/DiffSynth-Studio
pip install -e .
```

## 🛠️ Evaluation on IF-Bench
All supported models are listed in `load_func_dict` in `evaluate/bench_evaluate.py`. You can add your own model by adding a new model loading function in `load_func_dict` and its corresponding infer function. We provide some examples below.

### Quick Start
- Image Download

(1) Download images in IF-Bench and save in `if_bench/infrared_imgs`.

(2) Download translated images in GenViP and save in `if_bench/translated_rgb_imgs`.

- Launch Evaluation
```
cd ./evaluate

# evaluate qwen25_vl_7b
CUDA_VISIBLE_DEVICES=0 python3 bench_evaluate.py \
    --model_name qwen25_vl_7b \ 
    --bench_file if_bench/if_bench.json \
    --img_dir_base_path if_bench/infrared_imgs \
    --save_dir /path/to/save/dir \
    --recycle_test \
    2>&1 | tee /path/to/log/dir/qwen25_vl_7b.log


# evaluate internvl35_8b with thinking
CUDA_VISIBLE_DEVICES=0 python3 bench_evaluate.py \
    --model_name internvl35_8b \ 
    --bench_file if_bench/if_bench.json \
    --img_dir_base_path if_bench/infrared_imgs \
    --save_dir /path/to/save/dir \
    --recycle_test \
    --think_mode think \
    2>&1 | tee /path/to/log/dir/internvl35_8b_thinking.log


# evaluate qwen3_vl_235b_a22b_instruct
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 bench_evaluate.py \
    --model_name qwen3_vl_235b_a22b_instruct \ 
    --bench_file if_bench/if_bench.json \
    --img_dir_base_path if_bench/infrared_imgs \
    --save_dir /path/to/save/dir \
    --recycle_test \
    2>&1 | tee /path/to/log/dir/qwen3_vl_235b_a22b_instruct.log
```

- Launch Evalution with GenViP
```
cd ./evaluate

# evaluate qwen25_vl_7b
CUDA_VISIBLE_DEVICES=0 python3 bench_evaluate.py \
    --model_name qwen25_vl_7b \ 
    --bench_file if_bench/if_bench.json \
    --img_dir_base_path if_bench/infrared_imgs \
    --save_dir /path/to/save/dir \
    --recycle_test \
    --rgbt_pair \
    --rgbt_transed_img_path if_bench/translated_rgb_imgs \
    --use_prior 
    2>&1 | tee /path/to/log/dir/qwen25_vl_7b_genvip.log

# other cases are similar to qwen25_vl_7b
```

### ⏩ Parallel Evalution with Multi-Node and Multi-GPU
To accelerate the evaluation, we support parallel evaluation with multi-node multi-gpu. Some examples are shown below.
- Prepare hostfile
Build a hostfile with the format as follows.
```
ip1
ip2
ip3
...
```

- Parallel Evaluation
```
cd ./evaluate

# evaluate qwen25_vl_7b
# launch parallel evaluation
python3 launch_inference.py --hostfile /path/to/hostfile \
    --save-dir /path/to/save/dir \
    --bench-file if_bench/if_bench_flatten_shuffle.json \
    --img-dir-base-path if_bench/infrared_imgs \
    --gpus-per-task 1 \
    --model-name qwen25_vl_7b \
    --recycle-test

# merge results when parallel evaluation is done
python3 merge_results.py \
    --result_dir /path/to/save/dir \
    --save_prefix qwen25_vl_7b_recycle


# evaluate internvl35_8b with thinking
python3 launch_inference.py --hostfile /path/to/hostfile \
    --save-dir /path/to/save/dir \
    --bench-file if_bench/if_bench_flatten_shuffle.json \
    --img-dir-base-path if_bench/infrared_imgs \
    --gpus-per-task 1 \
    --model-name internvl35_8b \
    --think-mode think \
    --recycle-test

# merge results when parallel evaluation is done
python3 merge_results.py \
    --result_dir /path/to/save/dir \
    --save_prefix internvl35_8b_thinking_recycle


# evaluate qwen3_vl_235b_a22b_instruct
python3 launch_inference.py --hostfile /path/to/hostfile \
    --save-dir /path/to/save/dir \
    --bench-file if_bench/if_bench_flatten_shuffle.json \
    --img-dir-base-path if_bench/infrared_imgs \
    --gpus-per-task 8 \
    --model-name qwen3_vl_235b_a22b_instruct \
    --recycle-test

# merge results when parallel evaluation is done
python3 merge_results.py \
    --result_dir /path/to/save/dir \
    --save_prefix qwen3_vl_235b_a22b_instruct_recycle
```

- Parallel Evaluation with GenViP
```
cd ./evaluate

# evaluate qwen25_vl_7b
# launch parallel evaluation
python3 launch_inference.py --hostfile /path/to/hostfile \
    --save-dir /path/to/save/dir \
    --bench-file if_bench/if_bench_flatten_shuffle.json \
    --img-dir-base-path /path/to/img/dir/of/IF-Bench \
    --gpus-per-task 1 \
    --model-name qwen25_vl_7b \
    --recycle-test \
    --rgbt-pair \
    --rgbt-transed-img-path if_bench/translated_rgb_imgs \
    --use-prior 

# merge results when parallel evaluation is done
python3 merge_results.py \
    --result_dir /path/to/save/dir \
    --save_prefix qwen25_vl_7b_recycle_rgbt_prior

# other cases are similar to qwen25_vl_7b
```

## ✈️ (Optional) Fine-tuning of Editing Models in GenViP
Our fine-tuning of Qwen-Edit-2509 is based on the DiffSynth-Studio.
Prepare training data `rgbt_dataset.json` following the format of `edit_ft/DiffSynth-Studio/data/example.json`.
### Fine-tuning of Qwen-Edit-2509
```
cd edit_ft/DiffSynth-Studio

accelerate launch --dynamo_backend no --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 --num_machines 1 \
  --main_process_port 29520 examples/qwen_image/model_training/train.py \
  --dataset_base_path /your/path/to/img_base_dir \
  --dataset_metadata_path /your/path/to/rgbt_dataset.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --edit_prompt_idx 1 \
  --height 1024 \
  --width 1024 \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./output/train_results/Qwen-Image-Edit-2509_lora32_bs8_1k_5w" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
```

### Inference
- Inference of Qwen-Edit-2509
```
cd edit_ft/DiffSynth-Studio

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 examples/qwen_image/model_training/validate_lora/qwen_image_edit_2509_multigpu.py \
    --model_name "origin_Qwen-Image-Edit-2509" \
    --epoch 2 \
    --edit_prompt_idx 1 \
    --test_json examples/qwen_image/model_training/validate_lora/if_bench_image.json \
    --src_dir /path/to/images/in/if_bench \
    --save_path /path/to/save/dir \
    --inference_step 40 \
    --height 1024 \
    --width 1024
```

- Inference of Qwen-Edit-2509-FT
```
cd edit_ft/DiffSynth-Studio

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 examples/qwen_image/model_training/validate_lora/qwen_image_edit_2509_multigpu.py \
    --model_name "Qwen-Image-Edit-2509_lora32_bs8_1k_50k" \
    --model_path /path/to/model/ckpt/ \
    --epoch 2 \
    --edit_prompt_idx 1 \
    --test_json examples/qwen_image/model_training/validate_lora/if_bench_image.json \
    --src_dir /path/to/images/in/if_bench \
    --save_path /path/to/save/dir \
    --inference_step 40 \
    --height 1024 \
    --width 1024
```

## ⭐ Citation 
If you find this repository helpful, please consider giving it a star ⭐.
