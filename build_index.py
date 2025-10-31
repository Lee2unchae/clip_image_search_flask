import os, numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# 경로 및 모델 설정
IMAGES_DIR = "static/images" # 이미지를 모아둔 폴더
OUT_PATH = "data/index.npz" # 임베딩 결과를 저장할 파일 경로
MODEL_ID = "openai/clip-vit-base-patch32" # 사용할 CLIP 모델 이름

# 이미지 불러오기 함수
def load_images(image_dir):
    files, imgs = [], []
    # 폴더 내 모든 파일 탐색
    for f in sorted(os.listdir(image_dir)):
        # 확장자가 이미지 파일인 경우만 선택
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            path = os.path.join(image_dir, f)
            try:
                # 이미지를 열고 RGB로 변환 (흑백 등 형식 통일)
                imgs.append(Image.open(path).convert("RGB"))
                files.append(f)
            except:
                # 이미지가 깨졌거나 열 수 없으면 건너뜀
                pass
    # 파일명 리스트, 이미지 객체 리스트 반환
    return files, imgs

# 메인 함수
def main():
    # data 폴더가 없으면 자동 생성
    os.makedirs("data", exist_ok=True)
    
    # 이미지 파일 로드
    files, imgs = load_images(IMAGES_DIR)

    # CLIP 모델 및 전처리기 불러오기
    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    #이미지 임베딩 계산
    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        img_features = model.get_image_features(**inputs)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        emb = img_features.cpu().numpy().astype("float32")

    # 결과 저장
    np.savez_compressed(OUT_PATH, emb=emb, files=np.array(files))
    print(f"✅ Saved {len(files)} embeddings to {OUT_PATH}")

if __name__ == "__main__":
    main()
