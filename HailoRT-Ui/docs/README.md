# Hailo-H8 Control Panel

Hailo-8 AI 가속기를 위한 Qt5 기반 GUI 애플리케이션입니다.

## 주요 기능

- **장치 모니터링**: Hailo-8 장치 상태, 온도, 전력, 사용률 실시간 모니터링
- **모델 변환**: PyTorch (.pt) → ONNX → HEF 변환 파이프라인
- **실시간 추론**: 카메라/비디오/이미지 입력으로 객체 감지 실행
- **성능 분석**: FPS, 지연시간, 처리량 실시간 모니터링

## 시스템 요구사항

- Python 3.8+
- Ubuntu 20.04+ / Windows 10+ (WSL2)
- Hailo-8 PCIe 가속기 (선택)
- HailoRT 4.x (선택)

## 빠른 시작

```bash
# 1. 의존성 설치
cd /home/life/Project/Hailo-H8/HailoRT-Ui
pip install -r requirements.txt

# 2. 애플리케이션 실행
python main.py
```

## 프로젝트 구조

```
HailoRT-Ui/
├── main.py                 # 애플리케이션 진입점
├── config.yaml             # 설정 파일
├── requirements.txt        # Python 의존성
│
├── ui/                     # Qt UI 파일 (.ui)
│   ├── main_window.ui
│   ├── tabs/
│   │   ├── device_tab.ui
│   │   ├── convert_tab.ui
│   │   ├── inference_tab.ui
│   │   └── monitor_tab.ui
│   └── dialogs/
│       └── settings.ui
│
├── src/                    # Python 소스 코드
│   ├── app.py              # 메인 애플리케이션
│   ├── views/              # UI 컨트롤러
│   ├── services/           # 비즈니스 로직
│   ├── workers/            # 백그라운드 작업
│   └── utils/              # 유틸리티
│
├── scripts/                # CLI 스크립트
│   ├── convert_pt_to_onnx.py
│   ├── compile_to_hef.py
│   └── prepare_calibration.py
│
└── docs/                   # 문서
```

### 데이터셋 구조 (권장)

데이터셋은 별도 폴더에 다음 구조로 구성:

```
<dataset-folder>/
├── models/                 # 모델 파일
│   ├── pt/                 # PyTorch (.pt)
│   ├── onnx/               # ONNX (.onnx)
│   ├── har/                # Hailo Archive (.har)
│   └── hef/                # Hailo Executable (.hef)
├── train/images/           # 학습 이미지
├── valid/images/           # 검증 이미지
├── test/images/            # 테스트 이미지
├── inference_output/       # 추론 결과
│   ├── images/
│   ├── videos/
│   └── results/
├── configs/                # 설정 파일
└── logs/                   # 로그
```

## 문서

- [설치 가이드](INSTALLATION.md)
- [사용자 가이드](USER_GUIDE.md)
- [API 문서](API.md)
- [문제 해결](TROUBLESHOOTING.md)

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.
