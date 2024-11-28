import torch
import clip
from PIL import Image
import os
import itertools
import torch.nn as nn

positive_root = './postive_examples/'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print(device)

dataset_folder = './coco_data'

images = []
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith(('jpg','jpeg')):
            images.append(  root  + '/'+ file)


#Embedding of the input image
# input_image = preprocess(Image.open("2.jpeg")).unsqueeze(0).to(device)
# input_image_features = model.encode_image(input_image)

result = {}
length = len(images)
i = 0
all_images_features = {}
for img in images:
    i=i+1
    print(f'{i}/{length}')
    with torch.no_grad():
        image_preprocess = preprocess(Image.open(img)).unsqueeze(0).to(device)
        image_features = model.encode_image( image_preprocess)
        all_images_features[img] = image_features
        # cos = torch.nn.CosineSimilarity(dim=0)
        # sim = cos(image_features[0],input_image_features[0]).item()
        # sim = (sim+1)/2
        # result[img]=sim
for folder in os.listdir(positive_root):
    print(folder)
    os.makedirs('./negative_examples/'+folder, exist_ok=True)
    for file in os.listdir(positive_root+folder):
        print(file)
        input_image = preprocess(Image.open(positive_root+folder+'/'+file)).unsqueeze(0).to(device)
        input_image_features = model.encode_image(input_image) 
        for img in all_images_features.keys():
            cos = torch.nn.CosineSimilarity(dim=0)
            sim = cos(all_images_features[img][0],input_image_features[0]).item()
            sim = (sim+1)/2
            result[img]=sim
        sorted_value = sorted(result.items(), key=lambda x:x[1], reverse=True)
        sorted_res = dict(sorted_value)
        top_20 = dict(itertools.islice(sorted_res.items(), 20))
        os.makedirs('./negative_examples/'+folder+'/'+file, exist_ok=True)
        for key in top_20.keys():
            os.system(f'cp -r {key} ./negative_examples/{folder}/{file}/')