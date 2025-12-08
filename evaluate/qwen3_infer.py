from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen3_7b():
    model_name = "Qwen/Qwen3-8B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda"
    )

    return model, tokenizer

def llm_check_infer(model, tokenizer, prompt):
    system_prompt = """
    You are a professional large model. You will receive responses from a model to single-choice questions. The options for each question are A, B, C, and D. Please determine which option (A, B, C, or D) the response considers to be the correct answer. If you believe that none of the options A, B, C, or D are present in the response, please reply with "F".
    Remember, you can only reply with one of the characters A, B, C, D, or F. Do not output anything else.
    Input: <input_response>
    """

    # prepare the model input
    prompt = system_prompt.replace("<input_response>", prompt)
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False 
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("thinking content:", thinking_content)
    # print("content:", content)
    return content


if __name__ == "__main__":
    prompt = "I think the answer is either A or B, and B seems more reasonable."
    model, tokenizer = load_qwen3_7b()
    response = llm_check_infer(model, tokenizer, prompt)
    print(response)