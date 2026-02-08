# 문제 해결 가이드

## 목차

1. [설치 문제](#설치-문제)
2. [장치 연결 문제](#장치-연결-문제)
3. [모델 변환 문제](#모델-변환-문제)
4. [추론 문제](#추론-문제)
5. [UI 문제](#ui-문제)
6. [성능 문제](#성능-문제)

---

## 설치 문제

### PyQt5 설치 실패

**증상**: `pip install PyQt5` 실패

**해결 방법**:

```bash
# Ubuntu/Debian
sudo apt-get install -y python3-pyqt5

# 또는 시스템 패키지 사용
sudo apt-get install -y pyqt5-dev-tools

# pip로 재시도
pip install PyQt5 --no-cache-dir
```

---

### OpenCV 설치 오류

**증상**: `ImportError: libGL.so.1: cannot open shared object file`

**해결 방법**:

```bash
# 필요한 라이브러리 설치
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# headless 버전 사용 (GUI 없는 환경)
pip uninstall opencv-python
pip install opencv-python-headless
```

---

### HailoRT 설치 실패

**증상**: `dpkg: error processing package hailort`

**해결 방법**:

```bash
# 의존성 해결
sudo apt-get install -f

# 강제 재설치
sudo dpkg -i --force-overwrite hailort_*.deb
```

---

## 장치 연결 문제

### 장치를 찾을 수 없음

**증상**: "No Hailo device found" 메시지

**확인 사항**:

1. **PCIe 연결 확인**:
   ```bash
   lspci | grep -i hailo
   # 출력 예: Hailo Technologies Ltd. Hailo-8 AI Processor
   ```

2. **드라이버 로드 확인**:
   ```bash
   lsmod | grep hailo
   # 출력 예: hailo_pci
   ```

3. **장치 파일 확인**:
   ```bash
   ls -la /dev/hailo*
   # 출력 예: /dev/hailo0
   ```

**해결 방법**:

```bash
# 드라이버 재로드
sudo modprobe -r hailo_pci
sudo modprobe hailo_pci

# 장치 권한 설정
sudo chmod 666 /dev/hailo0
# 또는 사용자를 hailo 그룹에 추가
sudo usermod -aG hailo $USER
```

---

### 권한 오류

**증상**: `Permission denied` 오류

**해결 방법**:

```bash
# udev 규칙 생성
sudo tee /etc/udev/rules.d/99-hailo.rules << EOF
SUBSYSTEM=="hailo", MODE="0666"
EOF

# udev 재로드
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

### WSL2에서 장치 인식 안됨

**증상**: WSL2에서 Hailo 장치가 보이지 않음

**설명**: WSL2는 기본적으로 PCIe 장치에 직접 접근할 수 없습니다.

**대안**:
1. Windows 네이티브 환경 사용
2. USB 연결 Hailo 장치 사용 (USB/IP를 통한 패스스루)
3. 더미/목(Mock) 모드로 UI 테스트

---

## 모델 변환 문제

### PT → ONNX 변환 실패

**증상**: `RuntimeError: ONNX export failed`

**확인 사항**:

1. **PyTorch 버전 호환성**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   # 권장: 1.9.0 이상
   ```

2. **모델 형식 확인**:
   ```python
   import torch
   model = torch.load("model.pt")
   print(type(model))
   # dict면 state_dict만 있을 수 있음
   ```

**해결 방법**:

```python
# state_dict만 있는 경우
# 모델 정의 코드가 필요합니다
from your_model import YourModel
model = YourModel()
model.load_state_dict(torch.load("model.pt"))
torch.save(model, "full_model.pt")
```

---

### YOLO 모델 변환 오류

**증상**: Ultralytics YOLO 모델 변환 실패

**해결 방법**:

```bash
# Ultralytics 최신 버전 설치
pip install ultralytics --upgrade

# Ultralytics의 내장 export 사용
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=11 simplify=True
```

---

### HEF 컴파일 실패

**증상**: `hailo_sdk_client` 오류

**확인 사항**:

1. **Hailo Dataflow Compiler 설치 확인**:
   ```bash
   python -c "from hailo_sdk_client import ClientRunner; print('OK')"
   ```

2. **캘리브레이션 데이터 확인**:
   ```bash
   ls -la data/calibration/images/
   # 최소 100개 이상의 이미지 필요
   ```

**일반적인 오류 및 해결**:

| 오류 | 원인 | 해결 |
|------|------|------|
| `No calibration data` | 캘리브레이션 이미지 없음 | 이미지 추가 |
| `Unsupported operation` | 지원하지 않는 연산자 | ONNX 모델 수정 필요 |
| `Memory allocation failed` | 메모리 부족 | 배치 크기 줄이기 |

---

### 지원하지 않는 ONNX 연산자

**증상**: `Unsupported ONNX operator: XXX`

**해결 방법**:

1. **opset 버전 변경**:
   ```bash
   python scripts/convert_pt_to_onnx.py model.pt --opset 13
   ```

2. **ONNX 모델 간소화**:
   ```bash
   pip install onnx-simplifier
   python -m onnxsim model.onnx model_simplified.onnx
   ```

3. **Hailo Model Zoo 참조**:
   지원되는 모델 아키텍처 확인

---

## 추론 문제

### 카메라가 열리지 않음

**증상**: "Failed to open video source" 오류

**해결 방법**:

```bash
# 카메라 장치 확인
ls -la /dev/video*

# 카메라 테스트
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"

# WSL2에서 카메라 사용 시
# Windows에서 usbipd로 USB 카메라 연결 필요
```

---

### FPS가 너무 낮음

**증상**: 예상보다 낮은 FPS

**확인 사항**:

1. **모델 최적화 확인**:
   - 컴파일 시 `--opt-level 2` 또는 `3` 사용

2. **입력 크기 확인**:
   - 작은 입력 크기 = 더 높은 FPS
   - 640x640 → 416x416 시도

3. **전처리/후처리 오버헤드**:
   - OpenCV 리사이즈 최적화
   - NumPy 연산 최적화

---

### 감지 결과가 이상함

**증상**: 잘못된 바운딩 박스 또는 클래스

**확인 사항**:

1. **신뢰도 임계값 조정**:
   - Tools → Settings → Confidence Threshold

2. **모델 입력 크기 일치**:
   ```python
   # 모델 학습 시 사용한 크기와 동일해야 함
   # 예: 640x640
   ```

3. **전처리 정규화**:
   - 0-255 vs 0-1 범위 확인
   - BGR vs RGB 순서 확인

---

## UI 문제

### 애플리케이션이 시작되지 않음

**증상**: `python main.py` 실행 시 오류

**일반적인 해결 방법**:

```bash
# PyQt5 재설치
pip uninstall PyQt5 PyQt5-Qt5 PyQt5-sip
pip install PyQt5

# 환경 변수 설정 (X11)
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb

# Wayland 사용 시
export QT_QPA_PLATFORM=wayland
```

---

### UI 파일을 찾을 수 없음

**증상**: `FileNotFoundError: .ui file not found`

**해결 방법**:

```bash
# UI 파일 존재 확인
ls -la ui/
ls -la ui/tabs/
ls -la ui/dialogs/

# 경로가 올바른지 확인
pwd  # HailoRT-Ui 폴더에서 실행해야 함
```

---

### 화면이 깨짐

**증상**: UI 요소가 제대로 표시되지 않음

**해결 방법**:

```bash
# High DPI 스케일링 비활성화
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_SCALE_FACTOR=1

python main.py
```

---

## 성능 문제

### 메모리 부족

**증상**: `MemoryError` 또는 시스템 느려짐

**해결 방법**:

1. **배치 크기 줄이기**
2. **캘리브레이션 이미지 수 줄이기**
3. **입력 해상도 낮추기**

```bash
# 시스템 메모리 확인
free -h

# 스왑 추가 (필요시)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

### 장치 온도 과열

**증상**: 온도가 70°C 이상

**해결 방법**:

1. **방열 확인**: 히트싱크/팬 상태 확인
2. **워크로드 감소**: 추론 중단 또는 FPS 제한
3. **전력 모드 조정**: 저전력 모드 사용

---

## 로그 수집

문제 보고 시 다음 정보를 포함하세요:

```bash
# 시스템 정보
uname -a
python --version
pip list | grep -E "PyQt5|torch|hailo|opencv"

# HailoRT 정보
hailortcli --version
hailortcli scan

# 로그 파일 (있는 경우)
cat ~/.hailo/logs/*.log
```

---

## 지원 요청

위 방법으로 해결되지 않는 경우:

1. GitHub Issues에 문제 보고
2. 다음 정보 포함:
   - 오류 메시지 전체
   - 재현 단계
   - 시스템 환경 (OS, Python 버전, 패키지 버전)
   - 로그 파일
