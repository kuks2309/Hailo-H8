# Hailo-Compiler-UI 설치 가이드

## 시스템 요구사항

### 하드웨어

| 항목 | 최소 | 권장 |
|------|------|------|
| RAM | 16GB | 32GB |
| CPU | x86_64 (AVX 지원) | - |
| GPU | 불필요 (CPU 사용) | CUDA GPU (선택) |
| 저장공간 | 10GB | 20GB+ |

### 소프트웨어

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 20.04 / 22.04 (WSL2 포함) |
| Python | 3.10 |
| Qt | 5.15+ |

---

## WSL2 설정 (Windows 사용자)

### 1. 메모리 할당

WSL2 기본 메모리는 부족할 수 있습니다. `.wslconfig`로 늘려주세요.

**Windows에서** `%USERPROFILE%\.wslconfig` 파일 생성:

```ini
[wsl2]
memory=12GB
processors=4
swap=4GB
```

적용:
```powershell
wsl --shutdown
```

### 2. 메모리 확인

```bash
free -h
```

---

## 시스템 패키지 설치

```bash
# 필수 패키지
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-tk

# Hailo Dataflow Compiler 의존성
sudo apt install -y graphviz graphviz-dev
```

---

## Python 가상환경 설정

### 1. 가상환경 생성

```bash
cd /home/life/Project/Hailo-H8
python3.10 -m venv hailo_env
```

### 2. 가상환경 활성화

```bash
source /home/life/Project/Hailo-H8/hailo_env/bin/activate
```

### 3. pip 업그레이드

```bash
pip install --upgrade pip
```

---

## Python 패키지 설치

### 필수 패키지

```bash
# PyQt5 (UI 프레임워크)
pip install PyQt5>=5.15.0

# 핵심 패키지
pip install numpy>=1.21.0
pip install Pillow>=9.0.0
pip install PyYAML>=6.0
pip install opencv-python>=4.5.0

# PyTorch (모델 변환)
pip install torch torchvision
```

### 선택 패키지

```bash
# ONNX 검증 (선택)
pip install onnx>=1.12.0

# YOLO 자동 감지 (선택)
pip install ultralytics>=8.0.0

# YOLOv5 (선택)
pip install yolov5
```

### Hailo SDK 패키지

```bash
# HailoRT (Python 바인딩)
pip install /home/life/Project/Hailo-H8/install_sw/hailort-4.23.0-cp310-cp310-linux_x86_64.whl

# Hailo Model Zoo
pip install /home/life/Project/Hailo-H8/install_sw/hailo_model_zoo-2.17.1-py3-none-any.whl

# Hailo Dataflow Compiler (HEF 컴파일용)
pip install /home/life/Project/Hailo-H8/install_sw/hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
```

### 한 번에 설치 (requirements.txt)

```bash
cd /home/life/Project/Hailo-H8/Hailo-Compiler-UI
pip install -r requirements.txt
```

---

## 의존성 목록

### requirements.txt

```
# Qt5 UI Framework
PyQt5>=5.15.0

# Core dependencies
numpy>=1.21.0
Pillow>=9.0.0
PyYAML>=6.0

# Image processing
opencv-python>=4.5.0

# PyTorch (for model conversion)
torch>=1.9.0
torchvision>=0.10.0

# ONNX (optional)
onnx>=1.12.0

# YOLO support (optional)
ultralytics>=8.0.0
```

### 패키지별 용도

| 패키지 | 용도 | 필수 |
|--------|------|------|
| PyQt5 | GUI 프레임워크 | ✅ |
| numpy | 배열/행렬 연산 | ✅ |
| Pillow | 이미지 처리 (캘리브레이션) | ✅ |
| PyYAML | 설정 파일 파싱 | ✅ |
| opencv-python | 이미지/비디오 처리 | ✅ |
| torch | PyTorch 모델 로딩 | ✅ |
| torchvision | 이미지 변환 유틸리티 | ✅ |
| onnx | ONNX 모델 검증 | ⚠️ 선택 |
| ultralytics | YOLO 모델 자동 감지/변환 | ⚠️ 선택 |
| hailort | Hailo 런타임 Python 바인딩 | ⚠️ 선택 |
| hailo_sdk_client | HEF 컴파일러 | ⚠️ 선택 |

---

## 설치 확인

### 환경 체크 스크립트

```bash
source /home/life/Project/Hailo-H8/hailo_env/bin/activate
cd /home/life/Project/Hailo-H8/Hailo-Compiler-UI

python -c "
from src.core.environment import check_environment

result = check_environment()
print('Environment Check:')
print(f'  All required OK: {result[\"all_required_ok\"]}')
print()
for pkg, info in result['packages'].items():
    status = '✓' if info.installed else '✗'
    version = info.version or 'N/A'
    req = '(required)' if info.required else '(optional)'
    print(f'  {status} {info.name}: {version} {req}')
"
```

### 예상 출력

```
Environment Check:
  All required OK: True

  ✓ torch: 2.x.x (required)
  ✓ numpy: 1.x.x (required)
  ✓ Pillow: x.x.x (required)
  ✓ ultralytics: 8.x.x (optional)
  ✓ hailo_sdk_client: 3.x.x (optional)
  ✓ onnx: 1.x.x (optional)
```

---

## 빠른 실행

### Bash Alias 설정

`~/.bashrc`에 추가:

```bash
# Hailo 프로젝트 바로가기
alias cdHailo="cd /home/life/Project/Hailo-H8"

# Hailo Compiler UI 실행
alias Hailo-CUI="source ~/Project/Hailo-H8/hailo_env/bin/activate && cd ~/Project/Hailo-H8/Hailo-Compiler-UI && python main.py"
```

적용:
```bash
source ~/.bashrc
```

### 실행

```bash
Hailo-CUI
```

---

## 문제 해결

### 1. python3-tk 미설치

```
[Error] Requirement python3-tk not found.
```

**해결:**
```bash
sudo apt install python3-tk
```

### 2. graphviz 헤더 없음

```
fatal error: graphviz/cgraph.h: No such file or directory
```

**해결:**
```bash
sudo apt install graphviz graphviz-dev
```

### 3. RAM 부족

```
[Error] The SDK requires 16 GB of RAM
```

**해결:** WSL2 메모리 증가 (위의 WSL2 설정 참조)

### 4. PyQt5 플랫폼 플러그인 오류

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```

**해결:**
```bash
sudo apt install libxcb-xinerama0 libxkbcommon-x11-0
```

### 5. Display 연결 안됨 (WSL2)

```
cannot connect to X server
```

**해결:** WSLg가 설치된 Windows 11 사용 또는 X서버 설치

---

## 디렉토리 구조

### 프로젝트 구조

```
/home/life/Project/Hailo-H8/
├── hailo_env/              # Python 가상환경
├── install_sw/             # Hailo 설치 패키지
│   ├── hailort-4.23.0-cp310-cp310-linux_x86_64.whl
│   ├── hailo_model_zoo-2.17.1-py3-none-any.whl
│   └── hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
├── Hailo-Compiler-UI/      # 컴파일러 UI (WSL2용)
├── HailoRT-Ui/             # 추론 UI
└── docs/                   # 프로젝트 문서
```

### 데이터셋 구조 (권장)

```
/home/life/Project/Yolov5_datasets/<dataset-name>/
├── data.yaml               # 데이터셋 설정
├── models/                 # 모델 파일
│   ├── pt/                 # PyTorch (.pt)
│   ├── onnx/               # ONNX (.onnx)
│   ├── har/                # Hailo Archive (.har)
│   └── hef/                # Hailo Executable (.hef)
├── train/images/           # 학습 이미지 (캘리브레이션용)
├── valid/images/           # 검증 이미지
├── test/images/            # 테스트 이미지
├── inference_output/       # 추론 결과
├── configs/                # 설정 파일
└── logs/                   # 로그
```

---

## 관련 문서

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 프로젝트 구조
- [Hailo Developer Zone](https://hailo.ai/developer-zone/) - 공식 문서
