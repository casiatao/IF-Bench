import os
# Install SDK:  pip install 'volcengine-python-sdk[ark]' .
from volcenginesdkarkruntime import Ark 
import base64




def load_seed_vision_client():
    api_key = "your_api_key"

    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3", 
        api_key=api_key, 
    )

    return client


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def seed_16_infer(client, system_prompt, img_path=None, max_tokens=1024):
    if img_path is None:
        completion = client.chat.completions.create(
            model = "doubao-seed-1-6-251015",
            messages = [
                {
                    "role": "user",  
                    "content": [   
                        {"type": "text", "text": system_prompt}, 
                    ],
                }
            ],
        )
    else:
        img_path = [img_path] if isinstance(img_path, str) else img_path

        completion = client.chat.completions.create(
            model = "doubao-seed-1-6-251015",
            messages = [
                {
                    "role": "user",  
                    "content": [   
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}} for img in img_path
                    ] + [
                        {"type": "text", "text": system_prompt}
                    ]
                }
            ],
        )

    return completion.choices[0].message.content


def seed_vision_16_infer(client, system_prompt, img_path=None, max_tokens=1024):
    if img_path is None:
        completion = client.chat.completions.create(
            model = "doubao-seed-1-6-vision-250815",
            messages = [
                {
                    "role": "user",  
                    "content": [   
                        {"type": "text", "text": system_prompt}, 
                    ],
                }
            ],
        )
    else:
        img_path = [img_path] if isinstance(img_path, str) else img_path

        completion = client.chat.completions.create(
            model = "doubao-seed-1-6-vision-250815",
            messages = [
                {
                    "role": "user",  
                    "content": [   
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}} for img in img_path
                    ] + [
                        {"type": "text", "text": system_prompt}
                    ]
                }
            ],
        )

    return completion.choices[0].message.content



if __name__ == "__main__":
    system_prompt = "Describe this image."
    img_path = [
        "example.png",
    ]
    output = seed_vision_16_infer(system_prompt, img_path)
    print(output)