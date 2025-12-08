from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info



def load_llava_onevision_15_4b_instruct():
    model_path = "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"

    # default: Load the model on the available device(s)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    return model, processor


def load_llava_onevision_15_8b_instruct():
    model_path = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

    # default: Load the model on the available device(s)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    return model, processor


def llava_onevision_15_infer(model, processor, system_prompt, img_path=None, max_new_tokens=1024):
    if img_path is None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]
    
    else:
        img_path = [img_path] if isinstance(img_path, str) else img_path
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img_path[i]} for i in range(len(img_path))] + 
                [
                    {"type": "text", "text": system_prompt},
                ]
            }
        ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


if __name__ == "__main__":
    model, processor = load_llava_onevision_15_4b_instruct()
    # model, processor = load_llava_onevision_15_8b_instruct()