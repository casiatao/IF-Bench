from transformers import AutoModel, AutoTokenizer, AutoProcessor
from keye_vl_utils import process_vision_info
import re

def load_keye_vl15_8b():
    # default: Load the model on the available device(s)
    model_path = "Kwai-Keye/Keye-VL-1_5-8B"

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        # flash_attention_2 is recommended for better performance
        attn_implementation="flash_attention_2",
    ).eval()

    model.to("cuda")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    return model, processor


def keye_vl15_infer_no_think(model, processor, system_prompt, img_path=None, max_new_tokens=1024):
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
                    {"type": "text", "text": system_prompt + "./no_think"},
                ]
            }
        ]

    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **mm_processor_kwargs
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    output_text = output_text[0]

    try:
        output_text = re.search(r"\\boxed{(.*?)}", output_text).group(1)
        output_text = output_text.strip()

    except:
        output_text = output_text

    
    return output_text



def keye_vl15_infer_auto(model, processor, system_prompt, img_path=None, max_new_tokens=1024):
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
    image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **mm_processor_kwargs
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text)

    output_text = output_text[0]

    if "</analysis>" in output_text:
        output_text = output_text.split("</analysis>")[-1].strip()

    try:
        output_text = re.search(r"\\boxed{(.*?)}", output_text).group(1)
        output_text = output_text.strip()

    except:
        output_text = output_text

    
    return output_text


def keye_vl15_infer_think(model, processor, system_prompt, img_path=None, max_new_tokens=1024):
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
                    {"type": "text", "text": system_prompt + "./think"},
                ]
            }
        ]

    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **mm_processor_kwargs
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output_text = output_text[0]
    # extract response from \\boxed{}
    try:
        output_text = re.search(r"\\boxed{(.*?)}", output_text).group(1)
        output_text = output_text.strip()

    except:
        output_text = output_text

    return output_text


if __name__ == "__main__":
    model, processor = load_keye_vl15_8b()