import torch
from torch.nn.functional import cross_entropy
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

class ImageTextDataset(Dataset):
    def __init__(self, image_paths, captions, processor, max_length=77):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")

        caption = self.captions[idx]
        processed = self.processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
        }


def load_data():
    captions_df = pd.read_csv(r'C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\captions_tokenized.csv')
    image_filenames = [os.path.join(r'C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\processed', fname) for fname in captions_df['image']]
    captions = captions_df['clean_caption'].tolist()
    return image_filenames, captions


def fine_tune_model():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to("cuda")

    image_filenames, captions = load_data()
    dataset = ImageTextDataset(image_paths=image_filenames, captions=captions, processor=processor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(4):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            optimizer.zero_grad()


            pixel_values = batch["pixel_values"].to("cuda")
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")


            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                return_loss=False,
            )


            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text


            batch_size = logits_per_image.size(0)
            labels = torch.arange(batch_size).to("cuda")


            loss_image = cross_entropy(logits_per_image, labels)
            loss_text = cross_entropy(logits_per_text, labels)
            loss = (loss_image + loss_text) / 2


            loss.backward()
            optimizer.step()

            total_loss += loss.item()


            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")


        if avg_loss < 0.01:
            print("Early stopping triggered.")
            break


    model.save_pretrained("models/fine_tuned_clip")
    processor.save_pretrained("models/fine_tuned_clip")
    print("Model fine-tuning completed and saved.")


if __name__ == "__main__":
    fine_tune_model()
