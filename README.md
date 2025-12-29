# 산업현장에서 작업자의 PPE 착용 여부 탐지 (KLGS)

> **팀명:** KLGS (Keep Laborers Guarded & Safe)  
> **프로젝트 기간:** 2025.10.17 ~ 2025.10.24  
> **주제:** 실시간 영상(이미지/동영상/웹캠)에서 작업자의 **PPE(개인보호장비) 착용/미착용 여부**를 탐지하는 객체탐지 기반 웹앱(Streamlit) 구현

---

## 0. 한 줄 요약

산업 현장에서 관리자가 육안으로 확인하던 PPE 착용 여부를, **YOLO + RT-DETR** 기반 객체 탐지로 자동화하고 **Streamlit 웹앱**으로 실시간 확인할 수 있게 만들었습니다.

---

## 1. 데모

### 1-1) 웹캠 실시간 데모 (영상 자리)


https://github.com/user-attachments/assets/10118e08-5901-4ba4-903f-97f08354e21f

roboflow : ppe-7lymj
https://app.roboflow.com/ddm-zpaft/ppe-7lymj/models


### 1-2) 결과 스냅샷

- 학습 결과 요약(예시)

![RT-DETR Training Results Overview](assets/images/training_results_rtdetr.png)

- Roboflow 라벨링/리라벨링 작업 화면(예시)

![Roboflow Annotation UI](assets/images/roboflow_annotation_ui.png)

- 추론 결과 비교(예시)

| YOLO 예시 | RT-DETR 예시 |
|---|---|
| ![YOLO inference example](assets/images/inference_yolo_example.jpg) | ![RT-DETR inference example](assets/images/inference_rtdetr_example.jpg) |

---

## 2. 문제 정의

### 2-1) 왜 필요한가?

- 건설/제조 등 산업 현장에서는 안전 규정이 존재하지만 **수칙 미준수**가 반복되고,
- PPE 미착용은 **시력/호흡기 손상 등 치명적 사고**로 이어질 수 있습니다.
- 기존에는 관리자의 육안 점검에 의존하여 **지속적·객관적 관리**에 한계가 있습니다.

### 2-2) 우리가 한 일

- CCTV/웹캠처럼 “지속적 스트림” 환경을 가정하고,  
  **실시간 추론이 가능한 객체탐지 모델**을 사용해 PPE 착용 여부를 탐지했습니다.
- 최종적으로 **웹앱(Streamlit)** 형태로 “현장 적용 가능한” 프로토타입을 만들었습니다.

---

## 3. 데이터셋 & 전처리

### 3-1) 원본 데이터

- 기본 베이스: Ultralytics Construction PPE 데이터셋
  - 특징: ‘미착용(No)’ 장비 클래스가 존재(예: no-helmet)

### 3-2) 원본 데이터 한계와 개선

- 원본 데이터는 **라벨 누락/오표기**가 존재하고, **데이터 수/품질**이 충분하지 않았습니다.
- 개선 내용
  - **클래스 재정의:** 예측 클래스 11개 → 7개로 축소 및 정렬
  - **리라벨링:** Roboflow로 누락 박스 추가/오표기 위치 수정
  - **데이터 증강:** 소량 클래스/난이도 케이스 보완 목적의 augmentation 적용

### 3-3) 최종 탐지 클래스 (7개)

- `person`
- `helmet`, `no_helmet`
- `goggles`, `no_goggles`
- `gloves`, `no_gloves`

> “미착용(no-*)” 클래스가 **현장에서 더 중요한 케이스**였고, 이 클래스들의 학습 난이도가 특히 높았습니다(아래 회고 참고).

---

## 4. 모델 선정

### 4-1) 선정 기준

- **실시간성(FPS)**이 핵심: 현장 적용을 위해 추론 속도가 중요
- CPU/MPS 환경에서도 “어느 정도” 동작 가능하도록 **경량/효율 모델 우선**
- 너무 무거운 ViT 기반/실시간성이 낮은 VLM 계열은 제외

### 4-2) 사용 모델

- **YOLO 계열 (주력 실시간 탐지 모델)**
  - 예: `YOLO11-nano`, `YOLO11-small`
  - 장점: 빠른 추론, 낮은 자원 사용, Streamlit 연동 용이

- **RT-DETR 계열 (속도-정확도 균형)**
  - 예: `RT-DETR v1 (l / xl)`, `RT-DETR v2 (s / m)` 실험
  - 장점: Transformer 기반 구조로 복잡한 배경/작은 객체에서도 강점 기대

> 참고: **RT-DETR v2**는 Ultralytics 인터페이스와 완전 결합되어 있지 않아(별도 레포 기반),  
> 본 프로젝트에서는 **Streamlit 연동 비용/시간 제약**으로 최종 웹앱 모델 후보에서 제외했습니다.

---

## 5. 학습 & 평가

### 5-1) 학습 파이프라인(요약)

1. 데이터셋 점검 → 클래스 재정의(11→7)
2. Roboflow 리라벨링(누락/오표기 박스 수정)
3. YOLO / RT-DETR 학습
4. Validation/Test 성능 비교 & 추론 예시 검증
5. Streamlit 웹앱에 모델 탑재(최종 산출물)

### 5-2) 성능 관찰(정성/정량)

- 착용 클래스 대비 **미착용(no-*) 클래스의 AP가 약 0.10~0.15 낮게** 관찰됨
- 원인 후보
  - **Class imbalance**
  - 유사한 구도/모델 반복 촬영으로 인한 데이터 다양성 부족
  - 데이터 도메인 편향(서양인/스톡 이미지 중심) → 한국 현장 이미지 일반화 한계 체감
  - gloves는 손 모양(펴짐/쥠), 가림(occlusion)에 민감

> 상세 수치/비교표는 별도 트래킹 문서(스프레드시트)에 정리했습니다.

---

## 6. Streamlit 웹앱

### 6-1) 제공 기능(최종)

- 이미지/동영상 업로드 기반 PPE 탐지
- (선택) 웹캠 실시간 PPE 탐지
- 결과 시각화: 바운딩 박스 + 클래스 표시

### 6-2) 추가로 넣고 싶었던 기능(아이디어)

- 현장별 필요한 PPE만 선택해서 탐지(checkbox)
- YOLO / RT-DETR 모델 선택 옵션
- 여러 샘플 영상을 미리 등록하고 선택 재생
- PPE 착용 조건을 더 직관적으로 표시(YES=초록 / NO=빨강 등)

### 6-3) 로컬 실행(예시)

> 실제 레포 구조에 맞춰 경로/파일명만 맞추면 됩니다.

```bash
# 1) 환경 구성
python -m venv .venv
source .venv/bin/activate  # (Windows는 .venv\Scripts\activate)
pip install -r requirements.txt

# 2) 실행
streamlit run app.py
```

---

## 7. (권장) 레포 구조

```text
.
├─ app.py                          # Streamlit 엔트리
├─ weights/
│  ├─ yolo_best.pt                 # TODO: YOLO 가중치
│  └─ rtdetr_best.pt               # TODO: RT-DETR(v1) 가중치
├─ data/
│  └─ ppe_relabelled/              # 리라벨링 완료 데이터
├─ notebooks/                      # EDA/실험 노트
├─ scripts/                        # 학습/평가/변환 스크립트
└─ assets/
   ├─ images/                      # README 이미지
   └─ videos/                      # README 데모 영상(TODO)
```

---

## 8. 팀 구성 & 역할

- **공통:** Construction PPE 데이터셋 리라벨링(박스 수작업 조정), 노션 정리
- **고현석:** RT-DETR v1 학습/튜닝, YOLO v11 베이스라인, 모델별 Val/Test 성능 트래킹
- **곽주영:** YOLO v11 학습/튜닝, Github 페이지 관리, EDA 보고서(1·2차), Streamlit(Windows) 설치/테스트
- **이기은:** 데이터 변환, RT-DETR v2 학습/튜닝
- **손혁재:** Streamlit 웹앱 제작, Roboflow 데이터셋 관리

---

## 9. 회고 & 다음 단계

- 미착용(no-*) 케이스 중심으로 데이터 추가(특히 no-goggles / no-gloves / no-helmet)
- 한국 산업현장 이미지 추가로 도메인 갭 완화
- gloves의 손 모양/가림(occlusion) 대응을 위한 데이터 다양화 및 후처리 개선
- RT-DETR v2의 웹앱 연동(후처리/시각화 파이프라인 정리) 재도전

---

## 10. Acknowledgements

- Dataset: Ultralytics Construction PPE
- Tools: Ultralytics, Roboflow, Streamlit
