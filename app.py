from flask import Flask, render_template, request
import os, numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

# -------------------- 모델 로드 --------------------
MODEL_ID = "openai/clip-vit-base-patch32"  # stable & fast
INDEX_PATH = "data/index.npz"
IMAGES_DIR = "static/images"

app = Flask(__name__)

clip_model = CLIPModel.from_pretrained(MODEL_ID)
clip_proc = CLIPProcessor.from_pretrained(MODEL_ID)

# -------------------- 인덱스 로드 --------------------
idx = np.load(INDEX_PATH, allow_pickle=True)
IMG_EMB = idx["emb"]
IMG_FILES = idx["files"]

# --------------------한글 인식율을 높이기 위해서 한글 -> 영문 매핑 --------------------
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

# -------------------- 유틸 함수 --------------------
def _extract_keywords(q: str) -> list[str]:
    q = q.strip().lower()
    hits = []
    for k, vs in KW_MAP.items():
        if k in q:
            hits += vs
    tokens = [t for t in q.replace(",", " ").split() if t.isalpha()]
    hits += tokens
    return list(dict.fromkeys(hits))

def _candidate_mask_from_filenames(keywords: list[str]) -> list[int]:
    if not keywords:
        return []
    cands = []
    for i, f in enumerate(IMG_FILES):
        name = (f.item() if hasattr(f, "item") else f).lower()
        if any(kw in name for kw in keywords):
            cands.append(i)
    return cands

def _make_text_prompts(query: str, keywords: list[str]) -> list[str]:
    base = query.strip()
    cands = {base, f"{base}, high quality photo", f"a photo of {base}"}
    for kw in keywords[:3]:
        cands.add(kw)
        cands.add(f"a photo of {kw}")
        cands.add(f"{kw}, high quality photo")
    return list(cands)

# -------------------- 검색 함수 --------------------
def search_by_text(query, top_k=None, min_score=0.25):
    keywords = _extract_keywords(query)
    prompts = _make_text_prompts(query, keywords)

    with torch.no_grad():
        t_inputs = clip_proc(text=prompts, return_tensors="pt", padding=True)
        t_feats = clip_model.get_text_features(**t_inputs)
        t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
        t_feat = t_feats.mean(dim=0)
        t_feat = t_feat / t_feat.norm()
    t = t_feat.cpu().numpy().astype("float32")

    # 파일명 기반 후보군
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

    results = [
        {
            "file": IMG_FILES[i].item() if hasattr(IMG_FILES[i], "item") else IMG_FILES[i],
            "score": float((IMG_EMB[i] @ t)),
        }
        for i in top_idx
    ]
    return results

# -------------------- 라우트 --------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None)

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    if not query:
        return render_template("index.html", results=[], error="검색어를 입력하세요.")
    results = search_by_text(query, top_k=None, min_score=0.25)
    return render_template("index.html", results=results, query=query)

# -------------------- 메인 실행부 --------------------
if __name__ == "__main__":
    # reloader 끄고, 포트 충돌 방지 위해 8000번 사용
    app.run(debug=False, host="127.0.0.1", port=8000, use_reloader=False)
