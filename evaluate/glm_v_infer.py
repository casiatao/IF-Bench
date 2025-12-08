from transformers import AutoProcessor, Glm4vMoeForConditionalGeneration, Glm4vForConditionalGeneration
import torch
import re

def load_glm_41v_9b_thinking():
    MODEL_PATH = "THUDM/GLM-4.1V-9B-Thinking"
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor


def load_glm_45v_106b_a12b_thinking():
    MODEL_PATH = "zai-org/GLM-4.5V"
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Glm4vMoeForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )

    return model, processor


def glm_v_infer(model, processor, system_prompt, img_path=None, max_new_tokens=1024):
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

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    try:
        output_text = re.search(r"<answer><\|begin_of_box\|>(.*?)<\|end_of_box\|></answer>", output_text).group(1)
        output_text = output_text.strip()

    except:
        output_text = output_text

    return output_text


if __name__ == "__main__":
    model, processor = load_glm_41v_9b_thinking()
    # model, processor = load_glm_45v_106b_a12b_thinking()