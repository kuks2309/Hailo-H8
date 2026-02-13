# 설치 가이드

## 빠른 설치 (자동)

프로젝트 루트에서 실행:

```bash
cd /path/to/Hailo-H8
bash setup-env-ubuntu2204.sh
```

이 스크립트가 시스템 패키지, Python 가상환경(`hailo_env`), 의존성, Hailo SDK를 자동으로 설치합니다.
HailoRT-Ui와 Hailo-Compiler-UI 모두 설치됩니다.

> WSL2 환경에서는 `bash setup-env-wsl2.sh`를 사용하세요.

---

## 수동 설치

아래는 수동 설치를 위한 상세 가이드입니다.

## 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [Python 환경 설정](#python-환경-설정)
3. [의존성 설치](#의존성-설치)
4. [Hailo SDK 설치](#hailo-sdk-설치)
5. [Qt Designer 설치](#qt-designer-설치)

---

## 시스템 요구사항

### 하드웨어
- CPU: x86_64 또는 ARM64
- RAM: 8GB 이상 권장
- Hailo-8 PCIe 가속기 (옵션)

### 소프트웨어
- OS: Ubuntu 20.04+, Ubuntu 22.04, Windows 10+ (WSL2)
- Python: 3.8, 3.9, 3.10
- CUDA: 불필요 (Hailo는 자체 NPU 사용)

---

## Python 환경 설정

### 가상환경 생성 (권장)

```bash
# 프로젝트 루트로 이동
cd Hailo-H8

# 가상환경 생성
python3 -m venv hailo_env

# 가상환경 활성화
source hailo_env/bin/activate

# pip 업그레이드
pip install --upgrade pip
```

---

## 의존성 설치

### 기본 의존성

```bash
cd Hailo-H8/HailoRT-Ui
pip install -r requirements.txt
```

### 개별 패키지 설치

```bash
# Qt5 UI 프레임워크
pip install PyQt5>=5.15.0

# 핵심 라이브러리
pip install numpy>=1.21.0
pip install Pillow>=9.0.0
pip install PyYAML>=6.0

# 영상 처리
pip install opencv-python>=4.5.0

# PyTorch (모델 변환용)
pip install torch>=1.9.0
pip install torchvision>=0.10.0

# ONNX
pip install onnx>=1.12.0
pip install onnxruntime>=1.12.0

# YOLOv8 지원 (선택)
pip install ultralytics>=8.0.0
```

---

## Hailo SDK 설치

### 1. HailoRT 설치 (런타임)

HailoRT는 Hailo 장치와 통신하기 위한 런타임 라이브러리입니다.

```bash
# x86_64 시스템
sudo dpkg -i install_sw/hailort_4.23.0_amd64.deb

# ARM64 시스템 (Raspberry Pi 등)
sudo dpkg -i install_sw/hailort_4.23.0_arm64.deb

# Python 바인딩
pip install install_sw/hailort-4.23.0-cp310-cp310-linux_x86_64.whl
```

### 2. PCIe 드라이버 설치

```bash
sudo dpkg -i install_sw/hailort-pcie-driver_4.23.0_all.deb

# 드라이버 로드
sudo modprobe hailo_pci

# 장치 확인
ls /dev/hailo*
```

### 3. Hailo Dataflow Compiler 설치 (모델 컴파일용)

```bash
pip install install_sw/hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
```

### 4. Hailo Model Zoo 설치 (선택)

```bash
pip install install_sw/hailo_model_zoo-2.17.1-py3-none-any.whl
```

### 설치 확인

```bash
# HailoRT 버전 확인
hailortcli --version

# Python 바인딩 확인
python -c "import hailo_platform; print(hailo_platform.__version__)"

# 장치 스캔
hailortcli scan
```

---

## Qt Designer 설치

UI 파일을 편집하려면 Qt Designer가 필요합니다.

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y qttools5-dev-tools
```

### PyQt5-tools (대안)

```bash
pip install pyqt5-tools
```

### 실행

```bash
# 시스템 Qt Designer
designer

# PyQt5-tools
pyqt5-tools designer
```

---

## 설치 확인

모든 설치가 완료되면 애플리케이션을 실행하여 확인합니다:

```bash
cd Hailo-H8/HailoRT-Ui
python main.py
```

### 테스트 체크리스트

- [ ] 애플리케이션 창이 열림
- [ ] 모든 탭이 표시됨 (Device, Model Convert, Inference, Monitor)
- [ ] Device 탭에서 Connect 버튼 동작
- [ ] Model Convert 탭에서 파일 브라우저 동작

---

## 다음 단계

설치가 완료되면 [사용자 가이드](USER_GUIDE.md)를 참조하세요.
