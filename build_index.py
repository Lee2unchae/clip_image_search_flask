import os, numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

IMAGES_DIR = "static/images"
OUT_PATH = "data/index.npz"
MODEL_ID = "openai/clip-vit-base-patch32"

def load_images(image_dir):
    files, imgs = [], []
    for f in sorted(os.listdir(image_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            path = os.path.join(image_dir, f)
            try:
                imgs.append(Image.open(path).convert("RGB"))
                files.append(f)
            except:
                pass
    return files, imgs

def main():
    os.makedirs("data", exist_ok=True)
    files, imgs = load_images(IMAGES_DIR)
    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        img_features = model.get_image_features(**inputs)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        emb = img_features.cpu().numpy().astype("float32")

    np.savez_compressed(OUT_PATH, emb=emb, files=np.array(files))
    print(f"✅ Saved {len(files)} embeddings to {OUT_PATH}")

if __name__ == "__main__":
    main()
