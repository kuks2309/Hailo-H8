# Hailo-Compiler-UI 프로젝트 구조

## 개요

WSL2 환경에서 Hailo 모델 컴파일을 위한 PyQt5 기반 GUI 애플리케이션입니다.

- **목적**: PT → ONNX → HEF 변환 파이프라인 UI
- **플랫폼**: Linux (WSL2)
- **Python**: 3.10+
- **UI Framework**: PyQt5

---

## 디렉토리 구조

```
Hailo-Compiler-UI/
├── main.py                 # 애플리케이션 진입점
├── config.yaml             # 설정 파일
├── requirements.txt        # Python 의존성
├── docs/                   # 문서
│   └── PROJECT_STRUCTURE.md
│
├── src/                    # 소스 코드
│   ├── __init__.py
│   │
│   ├── core/               # 핵심 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── exceptions.py   # 커스텀 예외 클래스
│   │   ├── environment.py  # 환경 체크 로직
│   │   └── converter.py    # 모델 변환 서비스
│   │
│   ├── ui/                 # UI 컴포넌트
│   │   ├── __init__.py
│   │   ├── styles.py       # 다크 테마 스타일시트
│   │   ├── log_panel.py    # 로그 출력 패널
│   │   ├── converter_panel.py  # 변환 UI 카드
│   │   └── main_window.py  # 메인 윈도우
│   │
│   ├── workers/            # 백그라운드 작업
│   │   ├── __init__.py
│   │   └── convert_worker.py  # QThread 변환 워커
│   │
│   └── utils/              # 유틸리티 (예약)
│       └── __init__.py
│
└── resources/              # 리소스 파일
    ├── icons/              # 아이콘
    └── images/             # 이미지
```

### 데이터셋 구조 (외부 폴더)

데이터셋은 프로젝트 외부에 별도로 구성하는 것을 권장합니다:

```
<dataset-folder>/           # 예: ~/Project/Yolov5_datasets/Jig-latch-segement/
├── data.yaml               # 데이터셋 설정 (클래스, 경로)
├── configs/                # 학습/변환 설정 파일
├── models/                 # 모델 파일
│   ├── pt/                 # PyTorch weights (.pt)
│   ├── onnx/               # ONNX models (.onnx)
│   ├── har/                # Hailo Archive (.har)
│   └── hef/                # Hailo Executable (.hef)
├── train/                  # 학습 데이터
│   ├── images/             # 이미지 (캘리브레이션용으로도 사용)
│   └── labels/             # 라벨
├── valid/                  # 검증 데이터
│   ├── images/
│   └── labels/
├── test/                   # 테스트 데이터
│   ├── images/
│   └── labels/
├── inference_output/       # 추론 결과
│   ├── images/             # 이미지 결과
│   ├── videos/             # 비디오 결과
│   └── results/            # JSON/CSV 결과
├── logs/                   # 로그 파일
└── scripts/                # 커스텀 스크립트
```

---

## 파일 상세 설명

### 루트 파일

| 파일 | 설명 |
|------|------|
| `main.py` | 애플리케이션 진입점. 환경 체크 후 메인 윈도우 실행 |
| `config.yaml` | 기본 설정값 (입력 크기, opset, 타겟 등) |
| `requirements.txt` | pip 의존성 목록 |

### src/core/ - 핵심 로직

| 파일 | 클래스/함수 | 설명 |
|------|------------|------|
| `exceptions.py` | `CompilerUIError` | 기본 예외 클래스 |
| | `EnvironmentError` | 환경 체크 실패 |
| | `ModelLoadError` | 모델 로딩 실패 |
| | `ExportError` | ONNX 내보내기 실패 |
| | `CompilationError` | HEF 컴파일 실패 |
| | `CalibrationError` | 캘리브레이션 오류 |
| `environment.py` | `check_environment()` | 필수 패키지 설치 확인 |
| | `can_compile_hef()` | HEF 컴파일 가능 여부 |
| | `can_detect_yolo()` | YOLO 자동 감지 가능 여부 |
| `converter.py` | `ModelConverter` | 변환 서비스 클래스 |
| | `.convert_pt_to_onnx()` | PT → ONNX 변환 |
| | `.compile_onnx_to_hef()` | ONNX → HEF 컴파일 |

### src/ui/ - UI 컴포넌트

| 파일 | 클래스 | 설명 |
|------|--------|------|
| `styles.py` | `DARK_THEME` | 다크 테마 스타일시트 |
| | `COLORS` | 색상 팔레트 딕셔너리 |
| `log_panel.py` | `LogPanel` | 로그 출력 및 진행률 표시 위젯 |
| `converter_panel.py` | `PtToOnnxPanel` | PT → ONNX 변환 UI 카드 |
| | `OnnxToHefPanel` | ONNX → HEF 컴파일 UI 카드 |
| `main_window.py` | `EnvironmentPanel` | 환경 상태 표시 패널 |
| | `MainWindow` | 메인 윈도우 (모든 컴포넌트 조합) |

### src/workers/ - 백그라운드 작업

| 파일 | 클래스 | 설명 |
|------|--------|------|
| `convert_worker.py` | `ConvertWorker` | QThread 기반 변환 워커 |

---

## 의존성 흐름

```
main.py
    └── src/ui/main_window.py
            ├── src/ui/converter_panel.py
            ├── src/ui/log_panel.py
            ├── src/ui/styles.py
            └── src/core/environment.py
                    │
                    ↓
            src/workers/convert_worker.py
                    │
                    ↓
            src/core/converter.py
                    │
                    ↓
            src/core/exceptions.py
```

**특징:**
- 단방향 의존성 (순환 없음)
- UI → Workers → Core 레이어 분리
- 예외는 최하위 레이어에서 정의

---

## 시그널/슬롯 구조

```
ConvertWorker (QThread)
    ├── progress(int)      → LogPanel.set_progress()
    ├── log(str)           → LogPanel.log_info()
    ├── error(str, str)    → MainWindow._on_error()
    └── finished(bool,str) → MainWindow._on_conversion_done()

PtToOnnxPanel
    └── convert_clicked(dict) → MainWindow._on_pt_to_onnx()

OnnxToHefPanel
    └── compile_clicked(dict) → MainWindow._on_onnx_to_hef()
```

---

## 테마 색상

| 이름 | 색상 | 용도 |
|------|------|------|
| Primary | `#1a237e` | 버튼, 선택 |
| Accent | `#ff6f00` | 강조, 진행률 |
| Background | `#121212` | 배경 |
| Surface | `#1e1e1e` | 카드 배경 |
| Success | `#4caf50` | 성공 메시지 |
| Warning | `#ff9800` | 경고 메시지 |
| Error | `#f44336` | 에러 메시지 |

---

## 실행 방법

```bash
# 가상환경 활성화
source ~/Project/Hailo-H8/hailo_env/bin/activate

# 실행
cd ~/Project/Hailo-H8/Hailo-Compiler-UI
python main.py

# 또는 alias 사용
Hailo-CUI
```

---

## 필수 패키지

| 패키지 | 용도 | 필수 |
|--------|------|------|
| PyQt5 | UI 프레임워크 | ✅ |
| torch | 모델 로딩/변환 | ✅ |
| numpy | 데이터 처리 | ✅ |
| Pillow | 이미지 처리 | ✅ |
| ultralytics | YOLO 자동 감지 | ⚠️ 선택 |
| onnx | ONNX 검증 | ⚠️ 선택 |
| hailo_sdk_client | HEF 컴파일 | ⚠️ 선택 |

---

## 관련 프로젝트

| 프로젝트 | 위치 | 설명 |
|----------|------|------|
| HailoRT-Ui | `../HailoRT-Ui/` | Windows용 추론/모니터링 UI |
| hailo_env | `../hailo_env/` | 공용 Python 가상환경 |
