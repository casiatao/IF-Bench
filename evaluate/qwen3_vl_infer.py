from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
import torch


def load_qwen3_vl_4b_instruct():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

    return model, processor


def load_qwen3_vl_4b_thinking():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Thinking",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Thinking")

    return model, processor


def load_qwen3_vl_8b_instruct():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    return model, processor


def load_qwen3_vl_8b_thinking():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Thinking",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")

    return model, processor



def load_qwen3_vl_30b_a3b_instruct():
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

    return model, processor


def load_qwen3_vl_30b_a3b_thinking():
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Thinking",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Thinking")

    return model, processor


def load_qwen3_vl_235b_a22b_instruct():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    return model, processor


def load_qwen3_vl_235b_a22b_thinking():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Thinking")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-235B-A22B-Thinking",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    return model, processor


def qwen3_vl_infer(model, processor, system_prompt, img_path=None, max_new_tokens=1024):
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
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0]
        # print(output_text)
        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()
        return output_text


if __name__ == "__main__":
    model, processor = load_qwen3_vl_4b_instruct()
    # model, processor = load_qwen3_vl_4b_thinking()
    # model, processor = load_qwen3_vl_8b_instruct()
    # model, processor = load_qwen3_vl_8b_thinking()
    # model, processor = load_qwen3_vl_30b_a3b_instruct()
    # model, processor = load_qwen3_vl_30b_a3b_thinking()
    # model, processor = load_qwen3_vl_235b_a22b_instruct()
    # model, processor = load_qwen3_vl_235b_a22b_thinking()

