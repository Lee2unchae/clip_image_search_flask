# CLIP 이미지 검색 사전

자연어를 입력하면 사전에 올려둔 이미지들 중 의미가 가까운 이미지를 검색하여 보여주는 프로그램입니다.  
- 모델: CLIP
- 프레임워크: FLASK
- 데이터: `static/images/`에 포함된 샘플 이미지
- 인덱스: `build_index.py`로 CLIP 이미지 임베딩 생성 → `data/index.npz` 저장

# 실행 방법
git clone https://github.com/Lee2unchae/clip_image_search_flask.git
cd clip_image_search_flask

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

