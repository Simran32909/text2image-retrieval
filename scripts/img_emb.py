import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import numpy as np

IMAGE_DIR = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\processed"
OUTPUT_EMBEDDINGS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\image_embeddings.npy"

model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()


image_embeddings=[]
image_names=[]
for image_file in tqdm(os.listdir(IMAGE_DIR)):
    image_path=os.path.join(IMAGE_DIR,image_file)
    image=Image.open(image_path).convert("RGB")
    inputs=processor(images=image,return_tensors="pt")
    with torch.no_grad():
        output=model.get_image_features(**inputs)
    embeddings=output.squeeze().numpy()
    image_embeddings.append(embeddings)
    image_names.append(image_file)

np.save(OUTPUT_EMBEDDINGS_PATH, np.array(image_embeddings))