import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
IMG_DIR= r"/static/images/flickr30k_images"
PROCESSED_DIR=r"C:/Users/simra/PycharmProjects/cross_domain_transfer_learning/data/processed"
os.makedirs(PROCESSED_DIR,exist_ok=True)

preprocess=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

def preprocess_img(image_path, save_path):
    try:
        img=Image.open(image_path).convert('RGB')
        resized_img=img.resize((224,224))
        resized_img.save(save_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    for image_name in tqdm(os.listdir(IMG_DIR)):
        input_path = os.path.join(IMG_DIR, image_name)
        output_path = os.path.join(PROCESSED_DIR, image_name)
        preprocess_img(input_path, output_path)