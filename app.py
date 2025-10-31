# Flask : 웹 서버를 만들기 위한 프레임워크
# numpy : 수치 계산 라이브러리
# torch : 딥러닝 연산 라이브러리
# transformers : CLIP 모델을 불러오기 위한 라이브러리
from flask import Flask, render_template, request 
import os, numpy as np
import torch 
from transformers import CLIPModel, CLIPProcessor 

# 모델 및 데이터 설정
MODEL_ID = "openai/clip-vit-base-patch32"  # 사용할 CLIP 모델 이름
INDEX_PATH = "data/index.npz" # 미리 계산된 이미지 임베딩 저장 파일
IMAGES_DIR = "static/images" # 실제 이미지가 들어 있는 폴더

app = Flask(__name__) # Flask 앱 생성

# CLIP 모델 불러오기
# 텍스트와 이미지를 벡터 형태로 바꿔주는 사전학습 모델
clip_model = CLIPModel.from_pretrained(MODEL_ID)
clip_proc = CLIPProcessor.from_pretrained(MODEL_ID)

# 이미지 벡터 데이터 불러오기
# 미리 만들어둔 이미지와 파일명 정보를 읽기
idx = np.load(INDEX_PATH, allow_pickle=True)
IMG_EMB = idx["emb"]
IMG_FILES = idx["files"]

# 한글 인식율을 높이기 위해서 한글 -> 영문 매핑 
# CLIP은 영어로 학습되었기 때문에, 한글 단어를 영어 단어로 바꿔 검색 정확도를 높인다.
KW_MAP = {
    "강아지": ["dog"],
    "새": ["bird"],
    "고양이": ["cat"],
    "여자": ["girl"],
    "물고기": ["fish"],
    "야경": ["night", "night skyline"],
    "강": ["river"],
    "에펠탑": ["eiffel", "EiffelTower"],
    "노을": ["sunset"],
    "태양": ["sun"],
    "달": ["moon"],
    "꽃": ["flower"],
    "동물": ["animal"],
    "자동차": ["car"],
    "집": ["house"],
    "카페": ["cafe"],
    "커피": ["coffee"],
    "음식": ["food"],
    "피자": ["pizza"],
    "식물": ["plant"],
    "케이크": ["cake"],
    "햄버거": ["hamburger", "burger"],
    "파스타": ["pasta"],
    "스테이크": ["steak"],
    "비행기": ["airplane"],
    "사막": ["desert"],
    "도시": ["city"],
    "산": ["mountain"],
    "숲": ["forest"],
    "바다": ["ocean", "sea"]
}

# 유틸 함수
# 1. 사용자의 문장에서 키워드 추출
def _extract_keywords(q: str) -> list[str]:
    q = q.strip().lower()
    hits = []
    # 입력 문장에 한글 키워드가 있으면 영어 단어로 바꿔 추가
    for k, vs in KW_MAP.items():
        if k in q:
            hits += vs
    tokens = [t for t in q.replace(",", " ").split() if t.isalpha()]
    hits += tokens
    return list(dict.fromkeys(hits))

# 2. 이미지 파일명에서 키워드가 포함된 후보만 미리 골라내기
def _candidate_mask_from_filenames(keywords: list[str]) -> list[int]:
    if not keywords:
        return []
    cands = []
    for i, f in enumerate(IMG_FILES):
        name = (f.item() if hasattr(f, "item") else f).lower()
        if any(kw in name for kw in keywords):
            cands.append(i)
    return cands
    
# 3. 입력 문장을 다양한 형태의 문장 프롬프트로 확장
def _make_text_prompts(query: str, keywords: list[str]) -> list[str]:
    base = query.strip()
    cands = {base, f"{base}, high quality photo", f"a photo of {base}"}
    for kw in keywords[:3]:
        cands.add(kw)
        cands.add(f"a photo of {kw}")
        cands.add(f"{kw}, high quality photo")
    return list(cands)

# 핵심 검색 함수
def search_by_text(query, top_k=None, min_score=0.25):
    # 1. 검색어에서 키워드 추출
    keywords = _extract_keywords(query)
    prompts = _make_text_prompts(query, keywords)

    # 2. CLIP으로 텍스트 벡터 생성
    with torch.no_grad():
        t_inputs = clip_proc(text=prompts, return_tensors="pt", padding=True)
        t_feats = clip_model.get_text_features(**t_inputs)
        t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
        t_feat = t_feats.mean(dim=0)
        t_feat = t_feat / t_feat.norm()
    t = t_feat.cpu().numpy().astype("float32")

    # 3. 파일명 기반 후보군이 있으면 그 안에서 먼저 검색
    cand_idx = _candidate_mask_from_filenames(keywords)

    if cand_idx:
        emb = IMG_EMB[cand_idx]
        scores = emb @ t
        order_local = np.argsort(-scores)
        filtered_local = [cand_idx[i] for i in order_local if float(scores[i]) >= float(min_score)]
        if top_k:
            top_idx = filtered_local[:top_k]
        else:
            top_idx = filtered_local
    else:
        scores = IMG_EMB @ t
        order = np.argsort(-scores)
        filtered = [i for i in order if float(scores[i]) >= float(min_score)]
        if top_k:
            top_idx = filtered[:top_k]
        else:
            top_idx = filtered

    # 4. 최종 결과 리스트 생성
    results = [
        {
            "file": IMG_FILES[i].item() if hasattr(IMG_FILES[i], "item") else IMG_FILES[i],
            "score": float((IMG_EMB[i] @ t)),
        }
        for i in top_idx
    ]
    return results

# Flask 웹 페이지 라우트 설정
# 기본 페이지
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None)

# 검색 요청 처리
@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    if not query:
        # 검색어가 비었을 경우 에러 메시지 출력
        return render_template("index.html", results=[], error="검색어를 입력하세요.")
    # 검색 실행
    results = search_by_text(query, top_k=None, min_score=0.25)
    # 결과를 HTML 페이지에 전달
    return render_template("index.html", results=results, query=query)

# 프로그램 실행
if __name__ == "__main__":
    # reloader 끄고, 포트 충돌 방지 위해 8000번 사용
    app.run(debug=False, host="127.0.0.1", port=8000, use_reloader=False)
