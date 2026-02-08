# API 문서

이 문서는 HailoRT-UI의 주요 클래스와 API를 설명합니다.

## 목차

1. [Services](#services)
   - [HailoService](#hailoservice)
   - [ConverterService](#converterservice)
2. [Views](#views)
3. [Utils](#utils)

---

## Services

### HailoService

Hailo 장치와의 통신을 담당하는 서비스 클래스입니다.

**파일**: `src/services/hailo_service.py`

#### 클래스 정의

```python
class HailoService:
    """Hailo 장치 통신 서비스"""
```

#### 메서드

##### `connect() -> Optional[Any]`

Hailo 장치에 연결합니다.

```python
service = HailoService()
device = service.connect()
if device:
    print("연결 성공")
```

**반환값**: 연결된 장치 객체 또는 `None`

---

##### `disconnect()`

장치 연결을 해제합니다.

```python
service.disconnect()
```

---

##### `get_device_info() -> Dict[str, Any]`

장치 정보를 반환합니다.

```python
info = service.get_device_info()
print(f"온도: {info['temperature']}°C")
```

**반환값**:
```python
{
    'device_name': str,      # 장치 이름
    'architecture': str,     # 아키텍처 (hailo8, hailo8l, ...)
    'serial': str,           # 시리얼 번호
    'firmware': str,         # 펌웨어 버전
    'driver': str,           # 드라이버 버전
    'temperature': float,    # 온도 (°C)
    'power': float,          # 전력 (W)
    'utilization': float     # 사용률 (%)
}
```

---

##### `load_model(hef_path: str) -> bool`

HEF 모델을 로드합니다.

```python
success = service.load_model("/path/to/model.hef")
```

**파라미터**:
- `hef_path`: HEF 파일 경로

**반환값**: 성공 여부

---

##### `infer(frame: np.ndarray) -> List[Dict[str, Any]]`

프레임에 대해 추론을 실행합니다.

```python
import cv2
frame = cv2.imread("image.jpg")
detections = service.infer(frame)

for det in detections:
    print(f"{det['class']}: {det['confidence']:.2f}")
```

**파라미터**:
- `frame`: BGR 이미지 (numpy 배열)

**반환값**: 감지 결과 리스트
```python
[
    {
        'bbox': [x1, y1, x2, y2],  # 바운딩 박스
        'confidence': float,        # 신뢰도 (0-1)
        'class': str                # 클래스 이름
    },
    ...
]
```

---

### ConverterService

모델 변환을 담당하는 서비스 클래스입니다.

**파일**: `src/services/converter_service.py`

#### 클래스 정의

```python
class ConverterService:
    """모델 변환 서비스 (PT → ONNX → HEF)"""
```

#### 메서드

##### `set_callbacks(progress_cb=None, log_cb=None)`

진행률 및 로그 콜백을 설정합니다.

```python
converter = ConverterService()
converter.set_callbacks(
    progress_cb=lambda p: print(f"Progress: {p}%"),
    log_cb=lambda m: print(m)
)
```

---

##### `convert_pt_to_onnx(...) -> bool`

PyTorch 모델을 ONNX로 변환합니다.

```python
success = converter.convert_pt_to_onnx(
    pt_path="model.pt",
    onnx_path="model.onnx",
    input_size=(640, 640),
    batch_size=1,
    opset_version=11
)
```

**파라미터**:
| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `pt_path` | str | - | PyTorch 모델 경로 |
| `onnx_path` | str | - | 출력 ONNX 경로 |
| `input_size` | Tuple[int, int] | (640, 640) | 입력 크기 (H, W) |
| `batch_size` | int | 1 | 배치 크기 |
| `opset_version` | int | 11 | ONNX opset 버전 |

**반환값**: 성공 여부

---

##### `compile_onnx_to_hef(...) -> bool`

ONNX 모델을 HEF로 컴파일합니다.

```python
success = converter.compile_onnx_to_hef(
    onnx_path="model.onnx",
    hef_path="model.hef",
    calib_dir="./calibration/images",
    target="hailo8",
    optimization_level=2
)
```

**파라미터**:
| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `onnx_path` | str | - | ONNX 모델 경로 |
| `hef_path` | str | - | 출력 HEF 경로 |
| `calib_dir` | str | - | 캘리브레이션 이미지 폴더 |
| `target` | str | "hailo8" | 대상 장치 |
| `optimization_level` | int | 2 | 최적화 레벨 (0-3) |

**반환값**: 성공 여부

---

##### `prepare_calibration_dataset(...) -> str`

캘리브레이션 데이터셋을 준비합니다.

```python
output_path = converter.prepare_calibration_dataset(
    input_dir="./images",
    output_path="./calibration.npy",
    input_size=(640, 640),
    max_images=500
)
```

**파라미터**:
| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `input_dir` | str | - | 이미지 폴더 경로 |
| `output_path` | str | - | 출력 .npy 파일 경로 |
| `input_size` | Tuple[int, int] | (640, 640) | 타겟 크기 |
| `max_images` | int | 500 | 최대 이미지 수 |

**반환값**: 저장된 파일 경로

---

## Views

### 탭 컨트롤러 구조

모든 탭 컨트롤러는 동일한 패턴을 따릅니다:

```python
class TabController:
    def __init__(self, tab_widget: QWidget, base_path: str):
        """
        Args:
            tab_widget: 탭 위젯
            base_path: 프로젝트 루트 경로
        """
        pass
```

### DeviceTabController

**파일**: `src/views/device_tab.py`

```python
controller = DeviceTabController(tab_widget, base_path)
controller.connect_device()      # 장치 연결
controller.disconnect_device()   # 장치 해제
controller.refresh_status()      # 상태 갱신
```

### ConvertTabController

**파일**: `src/views/convert_tab.py`

UI에서 변환 작업을 관리합니다. 내부적으로 `ConvertWorker` 스레드를 사용합니다.

### InferenceTabController

**파일**: `src/views/inference_tab.py`

```python
controller = InferenceTabController(tab_widget, base_path)
controller.load_model()          # 모델 로드
controller.unload_model()        # 모델 언로드
controller.start_inference()     # 추론 시작
controller.stop_inference()      # 추론 중지
```

### MonitorTabController

**파일**: `src/views/monitor_tab.py`

```python
controller = MonitorTabController(tab_widget, base_path)
controller.start_monitoring()    # 모니터링 시작
controller.stop_monitoring()     # 모니터링 중지
controller.update_metrics(fps, latency, throughput)  # 메트릭 업데이트
controller.reset_statistics()    # 통계 초기화
```

---

## Utils

### Config

**파일**: `src/utils/config.py`

설정 관리 싱글톤 클래스입니다.

```python
from utils.config import Config

# 설정 로드
config = Config("config.yaml")

# 값 읽기 (dot notation 지원)
threshold = config.get("inference.confidence_threshold", 0.5)

# 값 쓰기
config.set("inference.confidence_threshold", 0.6)

# 저장
config.save("config.yaml")
```

### Logger

**파일**: `src/utils/logger.py`

로깅 설정 유틸리티입니다.

```python
from utils.logger import setup_logger

logger = setup_logger("HailoRT-UI", log_dir="./logs")
logger.info("애플리케이션 시작")
logger.error("오류 발생")
```

---

## 확장하기

### 커스텀 서비스 추가

1. `src/services/` 에 새 파일 생성
2. `src/services/__init__.py` 에 import 추가
3. 필요한 곳에서 import하여 사용

### 커스텀 뷰 추가

1. `ui/` 에 .ui 파일 생성 (Qt Designer 사용)
2. `src/views/` 에 컨트롤러 클래스 생성
3. `src/app.py` 에서 컨트롤러 초기화

### 새로운 탭 추가

1. `ui/tabs/` 에 새 .ui 파일 생성
2. `ui/main_window.ui` 에 탭 추가
3. `src/views/` 에 탭 컨트롤러 생성
4. `src/app.py` 의 `_init_tabs()` 에서 초기화
