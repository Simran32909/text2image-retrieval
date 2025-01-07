from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model_path = r'C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\models\fine_tuned_clip'
processor = CLIPProcessor.from_pretrained(model_path)
model = CLIPModel.from_pretrained(model_path).to("cuda")

image_path = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\processed\36979.jpg"
test_captions = ["A dog playing with a ball", "A group of several men playing poker", "A person walking in the park"]

image = Image.open(image_path).convert("RGB")
inputs = processor(text=test_captions, images=image, return_tensors="pt", padding=True).to("cuda")

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print("Similarity scores:", probs)

image_path = f"C:/Users/simra/PycharmProjects/cross_domain_transfer_learning/data/images/flickr30k_images/{img}"
