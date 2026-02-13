# 사용자 가이드

## 목차

1. [애플리케이션 개요](#애플리케이션-개요)
2. [Device 탭](#device-탭)
3. [Model Convert 탭](#model-convert-탭)
4. [Inference 탭](#inference-탭)
5. [Monitor 탭](#monitor-탭)
6. [CLI 스크립트 사용법](#cli-스크립트-사용법)

---

## 애플리케이션 개요

Hailo-H8 Control Panel은 4개의 주요 탭으로 구성됩니다:

| 탭 | 기능 |
|----|------|
| **Device** | Hailo 장치 연결 및 상태 모니터링 |
| **Model Convert** | 모델 변환 (PT → ONNX → HEF) |
| **Inference** | 실시간 추론 실행 |
| **Monitor** | 성능 모니터링 |

### 실행 방법

```bash
cd Hailo-H8/HailoRT-Ui
python main.py
```

---

## Device 탭

Hailo-8 장치의 연결 상태와 실시간 정보를 모니터링합니다.

### 기능

- **연결 상태**: 장치 연결/해제
- **장치 정보**: 시리얼 번호, 펌웨어 버전, 드라이버 버전
- **실시간 모니터링**:
  - 온도 (°C)
  - 전력 소비 (W)
  - 사용률 (%)

### 사용 방법

1. **Connect** 버튼 클릭하여 장치 연결
2. 연결 성공 시 상태가 "● Connected" (녹색)으로 변경
3. 실시간 정보가 자동으로 업데이트됨
4. **Disconnect** 버튼으로 연결 해제

### 상태 표시

| 상태 | 설명 |
|------|------|
| ● Connected (녹색) | 장치 연결됨 |
| ● Disconnected (빨간색) | 장치 연결 안됨 |

### 온도 상태

| 온도 | 상태 | 색상 |
|------|------|------|
| < 50°C | 정상 | 녹색 |
| 50-70°C | 주의 | 주황색 |
| > 70°C | 경고 | 빨간색 |

---

## Model Convert 탭

PyTorch 모델을 Hailo HEF 형식으로 변환합니다.

### 변환 파이프라인

```
.pt (PyTorch) → .onnx (ONNX) → .hef (Hailo)
```

### Step 1: PyTorch → ONNX 변환

1. **PT File**: PyTorch 모델 파일 선택 (.pt)
2. **Input Size**: 모델 입력 크기 설정 (기본: 640x640)
3. **Batch Size**: 배치 크기 (기본: 1)
4. **Opset Version**: ONNX opset 버전 (권장: 11)
5. **Output**: 출력 ONNX 파일 경로
6. **Convert to ONNX** 버튼 클릭

### Step 2: ONNX → HEF 컴파일

1. **ONNX File**: 변환된 ONNX 모델 선택
2. **Calib Dir**: 캘리브레이션 이미지 폴더 선택
3. **Target**: 대상 장치 선택
   - `hailo8`: Hailo-8 (26 TOPS)
   - `hailo8l`: Hailo-8L (13 TOPS)
   - `hailo15h`: Hailo-15H
4. **Optimize**: 최적화 목표
   - `fps`: 프레임레이트 최적화
   - `accuracy`: 정확도 최적화
   - `power`: 전력 효율 최적화
5. **Compile to HEF** 버튼 클릭

### 캘리브레이션 데이터 준비

캘리브레이션에는 실제 추론할 데이터와 유사한 이미지가 필요합니다:

- **권장 이미지 수**: 100 ~ 500장
- **형식**: JPG, PNG
- **내용**: 실제 사용 환경과 유사한 이미지

```bash
# 데이터셋 폴더 구조 (권장)
<dataset-folder>/
├── models/
│   ├── pt/                 # PyTorch 모델
│   ├── onnx/               # ONNX 모델
│   ├── har/                # Hailo Archive
│   └── hef/                # Hailo Executable
├── train/
│   └── images/             # 학습 이미지 (캘리브레이션용)
├── valid/
│   └── images/             # 검증 이미지
└── test/
    └── images/             # 테스트 이미지
```

**캘리브레이션 이미지**: `train/images/` 또는 `valid/images/` 폴더 사용 권장

### 변환 로그

오른쪽 패널에서 변환 진행 상황과 로그를 확인할 수 있습니다.

---

## Inference 탭

로드된 HEF 모델로 실시간 추론을 실행합니다.

### 모델 로드

1. HEF 파일 경로 입력 또는 **...** 버튼으로 선택
2. **Load Model** 버튼 클릭
3. 로드 성공 시 모델 정보가 표시됨

### 입력 소스 설정

| 소스 타입 | 설명 | 예시 |
|----------|------|------|
| Camera | 웹캠/USB 카메라 | `0`, `1` |
| Video File | 비디오 파일 | `video.mp4` |
| Image File | 단일 이미지 | `image.jpg` |
| Image Folder | 이미지 폴더 | `./images/` |

### 추론 실행

1. **▶ Start** 버튼으로 추론 시작
2. 비디오 미리보기에서 결과 확인
3. **■ Stop** 버튼으로 중지
4. **📷 Capture** 버튼으로 현재 프레임 저장

### 결과 표시

- **Video Preview**: 바운딩 박스가 그려진 영상
- **Performance**: 실시간 FPS, 지연시간
- **Detection Results**: 감지된 객체 테이블 (클래스, 신뢰도, 개수)

---

## Monitor 탭

시스템 성능을 실시간으로 모니터링합니다.

### 표시 지표

| 지표 | 설명 |
|------|------|
| **FPS** | 초당 프레임 수 |
| **Latency** | 추론 지연시간 (ms) |
| **Throughput** | 처리량 (TOPS) |
| **Temperature** | 칩 온도 (°C) |

### 리소스 사용률

- **NPU Utilization**: Hailo NPU 사용률
- **Memory Usage**: 메모리 사용량
- **Power Consumption**: 전력 소비

### 기능

- **Auto Refresh**: 자동 갱신 토글
- **Refresh**: 수동 갱신
- **Export Data**: 성능 데이터 CSV 내보내기

---

## CLI 스크립트 사용법

GUI 없이 명령줄에서 작업할 수 있습니다.

### PT → ONNX 변환

```bash
python scripts/convert_pt_to_onnx.py <입력.pt> [옵션]

옵션:
  -o, --output    출력 파일 경로
  --size          입력 크기 (기본: 640 640)
  --batch         배치 크기 (기본: 1)
  --opset         ONNX opset 버전 (기본: 11)

예시:
  python scripts/convert_pt_to_onnx.py yolov8n.pt -o yolov8n.onnx --size 640 640
```

### 캘리브레이션 데이터 준비

```bash
python scripts/prepare_calibration.py <이미지_폴더> [옵션]

옵션:
  -o, --output      출력 파일 경로 (.npy)
  --size            이미지 크기 (기본: 640 640)
  --max-images      최대 이미지 수 (기본: 500)

예시:
  python scripts/prepare_calibration.py ./images -o calib.npy --max-images 300
```

### ONNX → HEF 컴파일

```bash
python scripts/compile_to_hef.py <입력.onnx> [옵션]

옵션:
  -o, --output      출력 파일 경로
  --calib           캘리브레이션 이미지 폴더 (필수)
  --target          대상 장치 (기본: hailo8)
  --opt-level       최적화 레벨 0-3 (기본: 2)

예시:
  python scripts/compile_to_hef.py yolov8n.onnx --calib ./calibration --target hailo8
```

---

## 설정

`config.yaml` 파일에서 기본 설정을 변경할 수 있습니다:

```yaml
paths:
  # 데이터셋 폴더 경로 (외부 폴더 지정 가능)
  dataset_root: /path/to/dataset
  models: ${dataset_root}/models
  calibration: ${dataset_root}/train/images
  output: ${dataset_root}/inference_output

inference:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  max_detections: 100

display:
  show_bboxes: true
  show_labels: true
  show_confidence: true
  show_fps: true
```

또는 **Tools → Settings** 메뉴에서 GUI로 설정할 수 있습니다.

### 모델 파일 경로 규칙

| 파일 타입 | 저장 위치 |
|----------|----------|
| PyTorch (.pt) | `models/pt/` |
| ONNX (.onnx) | `models/onnx/` |
| Hailo Archive (.har) | `models/har/` |
| Hailo Executable (.hef) | `models/hef/` |

---

## 다음 단계

- 문제가 발생하면 [문제 해결](TROUBLESHOOTING.md) 참조
- API 상세 정보는 [API 문서](API.md) 참조
