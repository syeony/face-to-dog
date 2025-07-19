# face-to-dog
사이드 프로젝트 (나와 닮은 동물은?)
# 실행명령어
alt+b로 터미널 열어서 명령어 순차적 입력
```
pip install -r requirements.txt
python app.py
```
# 원리
- HuggingFace의 사전학습된 Vision Transformer를 이용해 특징 벡터(embedding) 추출
- 각 강아지 이미지들도 특징 벡터로 변환
- 코사인 유사도(cosine similarity) 기반으로 가장 유사한 강아지 사진을 찾음
→ 즉, "가장 비슷한 방향"을 가진 특징 벡터를 찾는 것
핵심 비교 기준: Cosine Similarity
```python
from torch.nn.functional import cosine_similarity
similarity = cosine_similarity(user_embedding, dog_embedding, dim=1)
```
- 1.0에 가까울수록 완전히 닮은 것 (벡터 방향이 완전히 동일)
- 0에 가까우면 닮지 않음
# 화면
<img width="738" height="846" alt="image" src="https://github.com/user-attachments/assets/6d997dbc-526e-429a-b4cf-fc5732033bae" />
