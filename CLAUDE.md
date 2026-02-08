# Hailo-H8 Project

## Project Overview

PyQt5-based desktop application for managing Hailo-8 AI accelerators (NPUs).

**Stack:** Python 3.8+, PyQt5, PyTorch, OpenCV, ONNX, HailoRT

## Project Structure

```
Hailo-H8/
├── HailoRT-Ui/              # Main GUI application
│   ├── src/
│   │   ├── app.py           # Main window controller
│   │   ├── views/           # Tab controllers (MVC)
│   │   ├── services/        # Business logic layer
│   │   └── utils/           # Config, logging utilities
│   ├── ui/                  # Qt Designer .ui files
│   ├── scripts/             # CLI conversion tools
│   └── docs/                # Application documentation
│
├── Hailo-Compiler-UI/       # Model conversion tool (WSL2)
│   ├── src/                 # Source code
│   └── docs/                # Compiler documentation
│
├── hailo_env/               # Python virtual environment
├── install_sw/              # Hailo SDK installation packages
├── docs/                    # Project-wide documentation
└── CLAUDE.md                # This file
```

### Dataset Structure (Recommended)

프로젝트 외부에 데이터셋 폴더를 구성할 때 권장 구조:

```
<dataset-name>/
├── configs/                 # 학습/변환 설정 파일
├── data.yaml                # 데이터셋 설정 (클래스, 경로)
├── models/                  # 모델 파일
│   ├── pt/                  # PyTorch weights (.pt)
│   ├── onnx/                # ONNX models (.onnx)
│   ├── har/                 # Hailo Archive (.har)
│   └── hef/                 # Hailo Executable (.hef)
├── train/                   # 학습 데이터
│   ├── images/
│   └── labels/
├── valid/                   # 검증 데이터
│   ├── images/
│   └── labels/
├── test/                    # 테스트 데이터
│   ├── images/
│   └── labels/
├── inference_output/        # 추론 결과
│   ├── images/
│   ├── videos/
│   └── results/
├── logs/                    # 로그 파일
└── scripts/                 # 커스텀 스크립트
```

## Key Features

1. **Device Tab:** Connect/disconnect Hailo-8, monitor temperature/power/utilization
2. **Convert Tab:** PT → ONNX → HEF model conversion pipeline
3. **Inference Tab:** Real-time inference from camera/video/images
4. **Monitor Tab:** FPS, latency, NPU metrics with CSV export

## Development Notes

### Code Quality Standards

- Use specific exception types, not bare `except:`
- Use `torch.load(..., weights_only=False)` explicitly when full model needed
- Use logger from `utils/logger.py` instead of `print()`
- Follow existing MVC pattern: views in `views/`, logic in `services/`

### 이슈 및 수정사항 문서화 지침

이슈 발생 및 수정 시 [docs/ISSUE_FIX_LOG.md](docs/ISSUE_FIX_LOG.md)에 반드시 기록할 것.

**문서 작성 규칙:**
- 날짜, 이슈 설명, 원인 분석, 해결 방법을 명확히 기술
- 관련 파일 경로와 코드 변경 내용 포함
- 재발 방지를 위한 교훈 또는 권장사항 추가
- 심각도(Critical/Major/Minor) 표기

### Testing

- No test suite currently exists
- Priority: Add tests for `Config`, `ConverterService`, `HailoService`

### Dependencies

Required:
- PyQt5 >= 5.15.0
- PyTorch >= 1.9.0
- OpenCV >= 4.5.0
- ONNX >= 1.12.0

Optional:
- HailoRT (for actual device communication)
- Ultralytics (for YOLO model export)

## Commands

```bash
# Run application (from HailoRT-Ui directory)
python main.py

# Windows installation
install.bat   # or install.ps1

# Run application on Windows
run.bat       # or run.ps1
```

## Mock Mode

When HailoRT is not installed, the application runs in mock mode with simulated device data. This enables development and testing without physical Hailo hardware.
