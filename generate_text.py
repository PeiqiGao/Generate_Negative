from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import json
import os

json_save = {}

with open('./1.json','r') as f:
    raw = json.load(f)
questions = []
for element in raw['./data/train/mam/3.png']:
    questions.append(element['Human'])


processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True,device_map="auto")

# prepare image and text prompt, using the appropriate prompt template

for img in os.listdir('./postive_examples/peiqi'):
    print(img)
    json_save['./yollava-data/train/peiqi/'+img] = []
    image = Image.open('./postive_examples/peiqi/'+img)
    for question in questions:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=100)
        json_save['./yollava-data/train/peiqi/'+img].append({"Human":question, "AI":processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT:')[1]})
        print(processor.decode(output[0], skip_special_tokens=True))
with open('./text-only-conversation.json','w') as f:
    f.write(json.dumps(json_save, indent=4))