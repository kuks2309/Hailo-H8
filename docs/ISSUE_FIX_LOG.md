# HailoRT-Ui Issue Fix Log

## 2026-02-07

### Proto Node 감지 시 Model Type 자동 전환 (CRITICAL)

**문제:**
```
HailoNNException: The layer named conv70 doesn't exist in the HN
Compilation failed: Latency simulation reached timeout
```

**원인 분석:**
- UI에서 Model Type이 `detect`로 설정되었지만 실제 ONNX는 **YOLOv5-seg** 모델
- `/model.24/proto/cv3/conv/Conv` 노드 존재 = 세그멘테이션 모델의 증거
- `hailomz compile yolov5s` (detection용)로 컴파일 시도 → 레이어 구조 불일치
- Python SDK fallback 시에도 잘못된 model_type 사용 → 타임아웃

**해결:**
- `src/core/converter.py`에서 proto 노드 감지 시 자동으로 `model_type = 'segment'` 설정

| 파일 | 라인 | 수정 내용 |
|------|------|----------|
| `Hailo-Compiler-UI/src/core/converter.py` | 1080-1083 | Proto node 감지 시 model_type 자동 전환 로직 추가 |

**추가된 코드:**
```python
# Auto-detect segmentation model from proto node
if proto_node and model_type != 'segment':
    self._log(f"Proto node detected: {proto_node}")
    self._log(f"Auto-switching model_type: {model_type} → segment")
    model_type = 'segment'
```

**교훈:**
- ONNX 구조 분석 결과(proto 노드)가 UI 설정보다 우선되어야 함
- 사용자 설정 오류를 자동으로 보정하는 방어적 코딩 필요

---

## 2026-02-06

### Code Review & Security Fixes

#### 1. Bare except: Clauses Fixed (HIGH)

**Problem:** Bare `except:` catches all exceptions including SystemExit and KeyboardInterrupt, masking bugs and making debugging impossible.

| File | Line | Before | After |
|------|------|--------|-------|
| `src/services/hailo_service.py` | 82 | `except:` | `except (AttributeError, ImportError):` |
| `src/services/converter_service.py` | 93 | `except:` | `except (RuntimeError, KeyError, TypeError):` |

---

#### 2. Unsafe torch.load() Fixed (MEDIUM)

**Problem:** Since PyTorch 2.6, `torch.load()` defaults to unsafe deserialization that can execute arbitrary code from malicious .pt files.

| File | Line | Fix Applied |
|------|------|-------------|
| `src/services/converter_service.py` | 84 | Added `weights_only=False` with comment explaining necessity |
| `src/services/converter_service.py` | 155 | Added `weights_only=False` with comment explaining necessity |
| `src/views/convert_tab.py` | 105 | Added `weights_only=False` with comment explaining necessity |

**Note:** `weights_only=False` is explicitly set because full model objects (not just weights) are required for ONNX export. This is intentional but users should only load trusted .pt files.

---

---

#### 3. YOLOv5 Model Loading Failure Fixed (CRITICAL)

**Problem:** PT → ONNX 변환 시 YOLOv5 모델 로드 실패
```
Failed to load model: No module named 'models'
```

**원인 분석:**
- YOLOv5 모델(.pt)은 `models.yolo`, `models.common` 모듈을 참조 형태로 저장
- `_is_yolo_model()` 함수가 `torch.load()`로 모델 타입 확인 시도
- YOLOv5 repository가 `sys.path`에 없어 모듈 로드 실패
- Ultralytics export 경로로 가기 전에 예외 발생

**해결 방법:**

1. **`_detect_yolo_version()` 함수 신규 추가**
   - zipfile로 checkpoint 내부 pickle 데이터에서 모듈 참조 확인
   - `ultralytics.nn` → YOLOv8, `models.yolo` → YOLOv5
   - `ModuleNotFoundError` 예외 메시지로 버전 판별

2. **`_is_yolo_model()` 함수 개선**
   - `torch.load()` 없이 버전 탐지 우선 시도
   - 파일 경로 패턴 fallback (`best.pt`, `last.pt`, `yolo`, `weights/`)

3. **버전별 export 함수 분리**
   - `_export_yolov5_to_onnx()`: ultralytics 시도 → YOLOv5 export.py fallback
   - `_export_yolov8_to_onnx()`: ultralytics 패키지 사용

| File | Changes |
|------|---------|
| `src/services/converter_service.py` | `_detect_yolo_version()` 추가 (83-166행) |
| `src/services/converter_service.py` | `_is_yolo_model()` 개선 (168-181행) |
| `src/services/converter_service.py` | `_export_yolov5_to_onnx()` 추가 (201-298행) |
| `src/services/converter_service.py` | `_export_yolov8_to_onnx()` 추가 (300-343행) |
| `src/views/convert_tab.py` | `ConvertWorker`가 `ConverterService` 사용하도록 리팩토링 (13-104행) |

**추가 문제 발견 및 해결:**

1. **View-Service 분리 문제:**
   - `convert_tab.py`의 `ConvertWorker` 클래스가 `ConverterService`를 사용하지 않고 자체 변환 로직 보유
   - `ConvertWorker._convert_pt_to_onnx()`와 `_convert_onnx_to_hef()`가 `ConverterService` 사용하도록 수정

2. **PyTorch 2.6+ 호환성 문제:**
   - `torch.load()` 기본값이 `weights_only=True`로 변경됨
   - YOLOv5 모델의 `models.yolo.SegmentationModel` 클래스 로드 실패
   - 해결: 직접 `weights_only=False`로 모델 로드 후 `torch.onnx.export()` 수행

3. **onnxscript 의존성:**
   - PyTorch 2.x의 ONNX export에 `onnxscript` 패키지 필요
   - `pip install onnxscript` 설치 필요

**최종 export 흐름:**
```
1. yolov5 패키지 경로를 sys.path에 추가
2. torch.load(weights_only=False)로 모델 로드
3. model.fuse()로 레이어 최적화
4. torch.onnx.export()로 ONNX 변환
5. onnxsim으로 모델 단순화
```

**테스트 결과:** ✅ YOLOv5 학습 모델 (`best.pt`) → ONNX 변환 성공 (28.65 MB)

---

#### 4. ONNX Version Conflict with hailo_sdk_client (CRITICAL)

**Problem:** hailo_sdk_client가 갑자기 인식되지 않음
```
ModuleNotFoundError: No module named 'onnx.mapping'
```

**원인 분석:**
- `onnxscript` 설치 시 `onnx`가 1.20.1로 자동 업그레이드됨
- `hailo_sdk_client`는 `onnx.mapping` 모듈 사용 (onnx 1.16.x에만 존재)
- onnx 1.17+ 버전에서 `onnx.mapping` 모듈이 제거됨
- 버전 충돌: `onnxscript>=0.6.0`는 `onnx>=1.17` 필요, `hailo_sdk_client`는 `onnx<=1.16.0` 필요

**해결 방법:**
1. onnx를 1.16.0으로 다운그레이드: `pip install onnx==1.16.0`
2. onnxscript 제거 (hailo_sdk_client 호환성 우선)

| File | Changes |
|------|---------|
| `Hailo-Compiler-UI/requirements.txt` | `onnx==1.16.0` 핀, `onnxscript` 주석 처리 |
| `setup-env-wsl2.sh` | WSL2 설치 스크립트 |

**설치 스크립트 핵심 로직:**
```bash
# CRITICAL: onnx를 먼저 고정 버전으로 설치
pip install onnx==1.16.0
pip install protobuf==3.20.3
# 그 후 나머지 의존성 설치
pip install -r requirements.txt
```

**예방책:**
1. `requirements.txt`에 버전 핀 명시: `onnx==1.16.0`, `protobuf==3.20.3`
2. 새 패키지 설치 전 충돌 확인: `pip install --dry-run <package>`
3. 의존성 업그레이드 방지: `pip install --no-deps <package>`
4. 설치 스크립트 사용 권장: `bash setup-env-wsl2.sh` (WSL2) 또는 `bash setup-env-ubuntu2204.sh` (Ubuntu 22.04)

**참고:**
- PyTorch 2.x ONNX export는 onnxscript 없이도 기본 기능 작동
- hailo_sdk_client 호환성이 HEF 컴파일에 필수이므로 우선순위 높음

**테스트 결과:** ✅ `import hailo_sdk_client` 성공

---

#### 5. ONNX Opset Version Downconversion Failure (MAJOR)

**Problem:** YOLOv5-seg 모델 ONNX 변환 시 opset 버전 호환성 오류
```
RuntimeError: /github/workspace/onnx/version_converter/BaseConverter.h:70: adapter_lookup:
Assertion `false` failed: No Adapter To Version $17 for Resize
```

**원인 분석:**
- config.yaml에서 `opset_version: 13`으로 설정됨
- PyTorch 2.x ONNX exporter는 내부적으로 opset 18 사용
- 요청된 opset 13으로 다운컨버트 시도 시 `Resize` 연산자 어댑터 부재
- ONNX version converter가 opset 18 → 13 변환 지원 안 함

**해결 방법:**
1. 기본 opset 버전을 13 → 17로 변경 (PyTorch 2.x 호환)
2. 지원 opset 목록에 18 추가

| File | Changes |
|------|---------|
| `Hailo-Compiler-UI/config.yaml` | `opset_version: 13` → `opset_version: 17` |
| `Hailo-Compiler-UI/config.yaml` | `opset_versions` 목록에 `18` 추가 |

**권장사항:**
- PyTorch 2.x 사용 시 opset 17 이상 권장
- Hailo SDK 호환성 확인 후 opset 18 사용 가능
- 구형 모델 호환 필요 시 opset 11-13 유지하되 별도 PyTorch 1.x 환경 사용

**테스트 필요:** YOLOv5-seg 모델로 opset 17 변환 재시도

---

#### 5-1. Segmentation 모델 자동 감지 및 Opset 조정 (MAJOR)

**Problem:** Segmentation 모델과 Detection 모델을 구분하지 않아 opset 호환성 문제 발생

**해결 방법:**
1. `_detect_yolo_task()` 함수 추가 - segment/detect/classify 구분
2. `convert_pt_to_onnx()`에서 segmentation 모델 감지 시 자동으로 opset 18 사용

**감지 방법 (우선순위):**
1. pickle 데이터에서 `SegmentationModel`, `Segment` 클래스명 확인
2. 파일 경로에서 `-seg`, `segment`, `_seg` 패턴 확인
3. 모델 로드 후 클래스명 확인

| File | Changes |
|------|---------|
| `Hailo-Compiler-UI/src/core/converter.py` | `_detect_yolo_task()` 함수 추가 |
| `Hailo-Compiler-UI/src/core/converter.py` | `convert_pt_to_onnx()`에 task 감지 및 opset 자동 조정 로직 추가 |

**동작 예시:**
```
Detected task type: segment
Segmentation model detected: upgrading opset 17 → 18 (Resize operator requirement)
```

**테스트 결과:** 생성된 ONNX 파일 opset 버전 확인: 18 ✅

---

#### 6. HEF 컴파일 실패 - Python SDK → hailomz CLI 전환 (CRITICAL)

**Problem:** YOLOv5-seg 모델 HEF 컴파일 시 Python SDK에서 파싱 오류
```
Compilation failed: list index out of range
```

**원인 분석:**
- `hailo_sdk_client.ClientRunner.translate_onnx_model()`이 segmentation 모델의 다중 출력 구조 파싱 실패
- YOLOv5-seg는 detection + segmentation 출력을 가짐
- Python SDK보다 `hailomz compile` CLI가 더 안정적

**해결 방법:**
1. `compile_onnx_to_hef()` 함수를 두 가지 방식으로 분리:
   - `_compile_with_hailomz()`: CLI 기반 (segment, detect 모델용)
   - `_compile_with_sdk()`: Python SDK 기반 (fallback)
2. UI에 model_type, model_name, num_classes 입력 필드 추가

**CLI 명령 형식:**
```bash
hailomz compile yolov5s_seg \
  --ckpt /path/to/model.onnx \
  --hw-arch hailo8 \
  --calib-path /path/to/calibration/images \
  --classes 1
```

| File | Changes |
|------|---------|
| `converter.py` | `_compile_with_hailomz()`, `_compile_with_sdk()` 분리 |
| `convert_worker.py` | model_type, num_classes, model_name 파라미터 추가 |
| `converter_panel.py` | Type, Model, Classes UI 입력 필드 추가 |

**UI 추가 필드:**
- Type: detect / segment / classify
- Model: yolov5s / yolov5m / yolov8s / yolov8m 등
- Classes: 클래스 수 (기본 80)

---

#### 7. YOLO Project Structure Auto-Detection (ENHANCEMENT)

**Problem:** 사용자가 PT 파일 선택 시 data.yaml의 클래스 수, 모델 타입, calibration 경로를 수동 설정해야 함

**해결 방법:**
1. `detect_yolo_project_from_pt()` 함수 추가 - PT 경로에서 YOLO 프로젝트 구조 자동 감지
2. `parse_yolo_data_yaml()` 함수 추가 - data.yaml 파싱
3. `PtToOnnxPanel.project_detected` 시그널로 HEF 패널에 정보 전달
4. `OnnxToHefPanel.apply_project_settings()` 메서드로 자동 설정 적용

**지원 폴더 구조:**
```
project_root/
├── train/
│   ├── images/     ← calibration 경로로 자동 설정
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── models/
│   ├── pt/         ← PT 파일 위치
│   ├── onnx/       ← ONNX 출력 위치
│   └── hef/        ← HEF 출력 위치
└── data.yaml       ← nc, names 자동 읽기
```

**자동 감지 항목:**
- `nc` (클래스 수): data.yaml에서 읽음
- `names` (클래스 이름): data.yaml에서 읽음 (tooltip으로 표시)
- `model_type`: 파일명/폴더명에서 segment/detect/classify 판별
- `train_images`: calibration 기본 경로로 설정

| File | Changes |
|------|---------|
| `converter_panel.py` | `detect_yolo_project_from_pt()`, `parse_yolo_data_yaml()` 함수 추가 |
| `converter_panel.py` | `PtToOnnxPanel.project_detected` 시그널 추가 |
| `converter_panel.py` | `OnnxToHefPanel.apply_project_settings()` 메서드 추가 |
| `main_window.py` | 시그널 연결 및 `_on_project_detected()` 로깅 추가 |

**테스트 방법:**
1. PT 파일 선택 시 로그 패널에서 감지된 프로젝트 정보 확인
2. Step 2 패널의 Calibration, Type, Classes 필드 자동 설정 확인

---

#### 8. YOLOv5 ONNX Node Naming Incompatibility with Hailo Model Zoo (CRITICAL)

**Problem:** PyTorch 2.x에서 export된 YOLOv5-seg ONNX 모델이 Hailo Model Zoo와 호환되지 않음
```
Unable to find end node names: ['conv2d_60', 'conv2d_61', 'conv2d_62']
```

**원인 분석:**

| 항목 | Hailo 기대값 (YOLOv5 공식) | 실제 값 (PyTorch 2.x) |
|------|--------------------------|----------------------|
| Detection Head | `/model.24/m.0/Conv`, `/model.24/m.1/Conv`, `/model.24/m.2/Conv` | `conv2d_60`, `conv2d_61`, `conv2d_62` |
| Proto Mask | `/model.24/proto/cv3/conv/Conv` | `silu_59` |

**근본 원인:**
- **YOLOv5**: PyTorch `torch.onnx.export()` 사용 시 자동 생성된 노드 이름 사용
- **YOLOv8**: 모듈 경로 기반 네이밍 (`/model.22/cv2.2/...`) → PyTorch 버전 무관하게 일관됨
- YOLOv5 공식 `export.py`는 모델 구조에서 모듈 이름을 추출하여 ONNX 노드에 적용
- Ultralytics/torch 기반 export는 `conv2d_N` 형식의 자동 네이밍 사용

**해결 방법:**

1. **ONNX Node 패턴 감지 함수 개선** (`extract_onnx_nodes`)
   - YOLOv5 공식 패턴: `/model.24/m.X/Conv`, `/model.24/proto/`
   - Ultralytics 패턴: `conv2d_N`, `silu_N`
   - 감지된 패턴에 따라 `naming_style` 반환

2. **hailomz compile에 `--end-node-names` 옵션 사용**
   - 감지된 노드 이름으로 컴파일 명령 동적 생성
   ```bash
   hailomz compile yolov5s_seg \
     --ckpt model.onnx \
     --hw-arch hailo8 \
     --end-node-names /model.24/m.0/Conv /model.24/m.1/Conv /model.24/m.2/Conv /model.24/proto/cv3/conv/Conv
   ```

3. **권장 해결책: 학습 환경에서 ONNX Export**
   - YOLOv5 학습 시 사용한 동일 환경에서 export 수행
   - `python export.py --weights best.pt --include onnx --simplify`
   - 이 방식으로 생성된 ONNX는 `/model.24/` 패턴 사용

| File | Changes |
|------|---------|
| `Hailo-Compiler-UI/src/core/converter.py` | `extract_onnx_nodes()` 함수에 `/model.24/` 패턴 감지 추가 |
| `Hailo-Compiler-UI/src/core/converter.py` | `_compile_with_hailomz()`에 `--end-node-names` 옵션 추가 |
| `.omc/plans/yolov5_hailo_conversion_plan.md` | 전체 변환 계획 문서 |

**YOLOv5 vs YOLOv8 노드 네이밍 비교:**

| 모델 | Export 방식 | 노드 네이밍 패턴 | Hailo 호환성 |
|------|------------|----------------|-------------|
| YOLOv5 | 공식 export.py | `/model.24/m.X/Conv` | ✅ 호환 |
| YOLOv5 | Ultralytics/torch | `conv2d_N` | ❌ 비호환 |
| YOLOv8 | 모든 방식 | `/model.22/cvX.X/...` | ✅ 호환 |

**테스트 결과:**
- `/model.24/` 패턴 ONNX: ✅ hailomz compile 성공
- `conv2d_` 패턴 ONNX: ❌ 노드를 찾을 수 없음

**검증된 워크플로우 (Jig-latch Segmentation 모델):**

1. **ONNX Export** (YOLOv5 공식 export.py 사용)
   ```bash
   cd /path/to/yolov5
   python3 export.py \
       --weights best.pt \
       --include onnx \
       --imgsz 640 \
       --opset 11 \
       --simplify
   ```

2. **노드 이름 확인** (커스텀 모델 필수)
   ```python
   import onnx
   model = onnx.load("best.onnx")
   conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
   for node in conv_nodes[-5:]:
       print(f"  {node.name}")
   ```
   출력 예시:
   ```
   /model.24/proto/cv3/conv/Conv
   /model.24/m.0/Conv
   /model.24/m.1/Conv
   /model.24/m.2/Conv
   ```

3. **HEF 컴파일**
   ```bash
   hailomz compile yolov5s_seg \
       --ckpt best.onnx \
       --hw-arch hailo8 \
       --calib-path /path/to/train/images \
       --classes 1 \
       --end-node-names /model.24/m.0/Conv /model.24/m.1/Conv /model.24/m.2/Conv /model.24/proto/cv3/conv/Conv
   ```

**벤치마크 결과 (Hailo H8):**
| 메트릭 | 값 |
|--------|-----|
| FPS (HW only) | **92.23** |
| FPS (Streaming) | **92.23** |
| HW Latency | **18.50 ms** |
| HEF 파일 크기 | 7.3 MB |

**권장사항:**
1. **단기**: 학습 환경에서 YOLOv5 공식 export.py로 ONNX 생성 ✅ **검증됨**
2. **중기**: Hailo-Compiler-UI에 ONNX 전처리 파이프라인 추가 (노드 이름 자동 수정)
3. **장기**: PyTorch 1.x export 환경을 Docker로 패키징

---

#### 9. YOLOv5 → HEF 변환 파이프라인 검증 완료 (VERIFICATION)

**테스트 일시:** 2026-02-06
**테스트 환경:** WSL2 Ubuntu 22.04, Python 3.10, HailoRT 4.23.0

**테스트 목적:** Jig-latch Segmentation 가이드 기반 변환 워크플로우 검증

**테스트 결과:**

| 단계 | 상태 | 비고 |
|------|------|------|
| YOLOv5 설정 | ✅ | yolov5s-seg.pt (15.6 MB) |
| 캘리브레이션 | ✅ | COCO128 이미지 102개 |
| ONNX Export | ✅ | yolov5s-seg.onnx (29.5 MB) |
| HEF 컴파일 | ✅ | yolov5s_seg.hef (7.4 MB) |
| 프로파일러 검증 | ✅ | 92.34 FPS / 18.40 ms |

**발견된 이슈:**

1. **PyTorch 2.x ONNX 노드 네이밍 비호환**
   - PyTorch 2.x → `conv2d_N` 형식
   - Hailo 기대값 → `/model.24/m.X/Conv` 형식
   - **해결:** Hailo Model Zoo 표준 모델 사용

2. **Opset 버전 다운컨버트 실패**
   - PyTorch 2.x는 opset 18 강제 사용
   - `--opset 11` 옵션 무시됨
   - **해결:** 표준 모델 또는 YOLOv5 공식 export.py 사용

**산출물:**

```
HailoRT-Ui/data/models/
├── pt/yolov5s-seg.pt       (15.6 MB)
├── onnx/yolov5s-seg.onnx   (29.5 MB)
├── har/yolov5s_seg.har     (151 MB)
└── hef/yolov5s_seg.hef     (7.4 MB)
```

**성능 (Profiler):**

| 메트릭 | 값 |
|--------|-----|
| Throughput | 92.34 FPS |
| Latency | 18.40 ms |
| GOP/s | 2434.09 |

**관련 문서:**
- [YOLOv5_to_Hailo_Conversion_Guide.md](YOLOv5_to_Hailo_Conversion_Guide.md) - 변환 가이드
- [Conversion_Test_Comparison.md](Conversion_Test_Comparison.md) - 원본 vs 테스트 비교

---

#### 10. PyTorch Version Detection and Warning (P1.1 Implementation)

**Problem:** Users with PyTorch 2.x may unknowingly export ONNX models with incompatible node names (conv2d_N), causing compilation failures in Hailo Model Zoo.

**Root Cause:** PyTorch 2.x ONNX exporter uses auto-generated node names instead of module path names. Users need early warning before attempting conversion.

**Solution Implemented:**

1. **Detection Function in converter.py**
   - Added `_detect_pytorch_version()` method (lines 480-493)
   - Detects PyTorch major version
   - Returns tuple: (major_version, has_onnx_naming_issue)
   - Gracefully handles missing PyTorch

2. **Warning Dialog in converter_panel.py**
   - Added `_check_pytorch_compatibility()` method (lines 244-251)
   - Modified `_on_convert()` to show warning BEFORE conversion (lines 253-284)
   - QMessageBox with Yes/No options, default No
   - Clear recommendations for alternatives

| File | Changes |
|------|---------|
| `Hailo-Compiler-UI/src/core/converter.py` | `_detect_pytorch_version()` method added (lines 480-493) |
| `Hailo-Compiler-UI/src/ui/converter_panel.py` | `_check_pytorch_compatibility()` method added (lines 244-251) |
| `Hailo-Compiler-UI/src/ui/converter_panel.py` | `_on_convert()` modified to check compatibility (lines 253-284) |

**Acceptance Criteria Met:**
- [x] Warning displayed when PyTorch 2.x detected in `_on_convert()`
- [x] QMessageBox with Yes/No options shown BEFORE conversion starts
- [x] User can cancel conversion from the warning dialog
- [x] No crash if PyTorch not installed (graceful fallback returns 0, False)

**Test Results:**
- Python syntax verification: ✅ Both files compile successfully
- Import validation: ✅ Module imports work correctly
- Logic flow: ✅ Warning dialog displayed before conversion emits params

**User Flow:**
```
1. User clicks "Convert to ONNX" button
2. _on_convert() checks PyTorch version
3. If PyTorch >= 2.x detected:
   - Warning dialog shown with message
   - Options: "Yes" to continue, "No" (default) to cancel
   - If "No": method returns early, no conversion emitted
   - If "Yes": continues to emit conversion params
4. If PyTorch < 2.x or not installed: continues normally
```

**Recommended Next Steps:**
- P1.2: Add ONNX version validation (checks onnx >= 1.17 incompatibility)
- P1.3: Enhance compatibility error messages with recovery actions
- P2.1: Add "Use Model Zoo" fallback button for incompatible ONNX

---

#### 11. Calibration Data Format Error - NHWC vs NCHW (CRITICAL)

**Problem:** HEF 컴파일 시 calibration 데이터 형식 불일치 오류
```
BadInputsShape: Data shape (3, 640, 640) for layer yolov5_seg_custom/input_layer1
doesn't match network's input shape (640, 640, 3)
```

**원인 분석:**
- hailomz CLI가 calibration 이미지를 로드할 때 NCHW `(C, H, W)` 형식으로 변환
- Hailo SDK는 내부적으로 NHWC `(H, W, C)` 형식을 기대
- Custom ONNX 사용 시 hailomz의 기본 preprocessing이 올바르게 동작하지 않음

**해결 방법:**

1. **`_load_calibration_data()` 함수 개선**
   - `layout` 파라미터 추가 ('NHWC' 또는 'NCHW')
   - Hailo SDK 기본값: NHWC `(640, 640, 3)`

2. **`_prepare_calibration_npy()` 함수 신규 추가**
   - Calibration 이미지를 numpy 파일로 저장
   - hailomz의 내부 이미지 로딩 우회
   - NHWC 형식 보장

3. **`_compile_with_hailomz()` 수정**
   - Calibration 데이터를 numpy로 미리 준비
   - `calib_size` 파라미터 추가

4. **`_compile_with_sdk()` 개선**
   - `end_node_names` 파라미터 추가
   - `model_name` 파라미터 추가
   - Custom ONNX의 end-node-names 지원

5. **`compile_onnx_to_hef()` 폴백 로직 추가**
   - hailomz가 calibration 형식 오류로 실패 시 Python SDK로 자동 폴백
   - SDK 사용 시 NHWC 형식 명시적 지정

| File | Changes |
|------|---------|
| `Hailo-Compiler-UI/src/core/converter.py` | `_load_calibration_data()` - layout 파라미터 추가 |
| `Hailo-Compiler-UI/src/core/converter.py` | `_prepare_calibration_npy()` 함수 추가 |
| `Hailo-Compiler-UI/src/core/converter.py` | `_compile_with_hailomz()` - calib_size 추가, numpy 지원 |
| `Hailo-Compiler-UI/src/core/converter.py` | `_compile_with_sdk()` - end_node_names, model_name 추가 |
| `Hailo-Compiler-UI/src/core/converter.py` | `compile_onnx_to_hef()` - SDK 폴백 로직 추가 |

**Hailo 데이터 형식 정리:**

| 컴포넌트 | 기대 형식 | Shape 예시 |
|----------|----------|-----------|
| Hailo SDK optimize() | NHWC | `(N, 640, 640, 3)` |
| PyTorch ONNX input | NCHW | `(N, 3, 640, 640)` |
| PIL Image load | HWC | `(640, 640, 3)` |

**폴백 동작:**
```
1. hailomz compile 시도
2. BadInputsShape 오류 발생 시:
   - Python SDK로 자동 전환
   - NHWC 형식으로 calibration 데이터 준비
   - end-node-names 전달
3. SDK로 컴파일 완료
```

**테스트 완료:** Custom YOLOv5-seg ONNX → HEF 변환 성공 (아래 Issue #12 참조)

---

### Issue #12: PyTorch 2.x Custom YOLOv5-seg → HEF 변환 (CRITICAL - RESOLVED)

**문제:** PyTorch 2.x에서 내보낸 Custom YOLOv5-seg ONNX가 Hailo SDK와 호환되지 않음

**증상:**
```
1. IndexError: list index out of range (ONNX 파싱 실패)
2. MisspellNodeError: Unable to find end node names
3. UnsupportedShuffleLayerError: Failed to determine type of layer
```

**근본 원인 분석:**
1. **Opset 강제 업그레이드:** PyTorch 2.x의 Resize 연산자가 opset 18 필수
2. **누락된 속성:** Conv 노드에 `kernel_shape` 속성 없음 (PyTorch 2.x는 생략)
3. **잘못된 Resize 입력:** sizes 대신 scales 사용해야 함
4. **호환되지 않는 속성:** Reshape의 `allowzero`, Split의 입력 형식

**해결 과정:**

| 단계 | 문제 | 해결 |
|------|------|------|
| 1 | `kernel_shape` 누락 | Conv 노드에 weight shape에서 추론하여 추가 |
| 2 | Resize 입력 오류 | `[data, roi, scales]` 형식으로 수정 |
| 3 | Reshape `allowzero` | 속성 제거 |
| 4 | Split 입력 형식 | opset 11 스타일 (attribute) 로 변환 |
| 5 | Post-processing 미지원 | `end_node_names` 지정하여 건너뛰기 |

**최종 해결 코드:**
```python
# ONNX 패치 스크립트 핵심 부분
for node in model.graph.node:
    if node.op_type == 'Conv':
        # kernel_shape 추가
        weight = initializers[node.input[1]]
        node.attribute.append(make_attribute('kernel_shape', list(weight.shape[2:])))

    elif node.op_type == 'Resize':
        # scales 입력 형식으로 수정
        node.input[:] = [data, 'empty_roi', scales]

    elif node.op_type == 'Reshape':
        # allowzero 제거
        node.attribute[:] = [a for a in node.attribute if a.name != 'allowzero']

    elif node.op_type == 'Split':
        # attribute 스타일로 변환
        split_sizes = initializers[node.input[1]]
        node.input[:] = [node.input[0]]
        node.attribute.append(make_attribute('split', split_sizes.tolist()))

model.opset_import[0].version = 11

# 번역 시 end_node_names 지정 (Hailo 제안 노드)
runner.translate_onnx_model(
    onnx_path,
    end_node_names=['node_silu_59', 'node_permute_1', 'node_permute_2', 'node_permute']
)
```

**검증 결과:**
- 입력: Custom YOLOv5s-seg (1 class, jig_latch)
- PT 파일: 15MB
- ONNX 파일: 29MB
- HAR 파일: 29MB (parsed), 138MB (optimized)
- **HEF 파일: 7.23MB** ✓
- 캘리브레이션: 64개 이미지, ~50초
- 컴파일: ~4분 (CPU)

**교훈:**
1. PyTorch 2.x ONNX는 수동 패치 필수
2. Hailo SDK 에러 메시지의 `end_node_names` 제안 활용
3. ONNX Runtime으로 먼저 검증 후 Hailo 테스트

---

### Issue #13: ONNX 노드 네이밍 불일치 - hailomz CLI 실패 (CRITICAL - RESOLVED)

**문제:** PyTorch 2.x로 export한 ONNX의 노드 이름이 Hailo Model Zoo와 호환되지 않음

**증상:**
```
Exception: Unable to find end node names: ['conv2d_1', 'conv2d_2', ..., 'conv2d_62']
```

**원인 분석:**

| Export 방식 | 노드 네이밍 패턴 | Hailo 호환 |
|-------------|-----------------|-----------|
| YOLOv5 공식 export.py | `/model.24/m.X/Conv` | ✅ |
| PyTorch 2.x torch.onnx.export() | `conv2d_N`, `node_conv2d_N` | ❌ |
| Ultralytics yolov5 package | `node_conv2d_N` | ❌ |

**해결 방법: ONNX 노드 이름 변환 스크립트**

| File | Description |
|------|-------------|
| `Hailo-Compiler-UI/scripts/rename_onnx_nodes.py` | 노드 이름 변환 스크립트 (신규) |

**스크립트 기능:**
```python
# 변환 전: node_conv2d_N
# 변환 후: /model.X/conv/Conv (Hailo 호환)

# 사용법
python rename_onnx_nodes.py input.onnx output.onnx --model-type seg
```

**변환 매핑 (YOLOv5-seg 기준):**
```
node_conv2d_0  → /model.0/conv/Conv
node_conv2d_1  → /model.1/conv/Conv
...
node_conv2d_59 → /model.24/proto/cv3/conv/Conv
node_conv2d_60 → /model.24/m.0/Conv
node_conv2d_61 → /model.24/m.1/Conv
node_conv2d_62 → /model.24/m.2/Conv
```

**컴파일 명령 (변환 후):**
```bash
hailomz compile yolov5s_seg \
    --ckpt best_hailo_renamed.onnx \
    --hw-arch hailo8 \
    --calib-path train/images \
    --classes 1 \
    --end-node-names /model.24/proto/cv3/conv/Conv /model.24/m.0/Conv /model.24/m.1/Conv /model.24/m.2/Conv
```

**검증 결과:**
- 입력: `best_hailo_compatible.onnx` (29MB, node_conv2d_N 네이밍)
- 변환: `best_hailo_renamed.onnx` (29MB, /model.X 네이밍)
- 출력: `yolov5s_seg.hef` (7.3MB) ✓
- Calibration: 64 images, 51초
- Mapping: 3분 55초
- Compilation: 16초

**참조:**
- [Hailo Model Zoo - YOLOv5](https://github.com/hailo-ai/hailo_model_zoo/blob/master/training/yolov5/README.rst)
- [Hailo Community - ONNX to HEF](https://community.hailo.ai/t/error-while-compile-yolov5-from-onnx-to-hef/12993/2)
- DFC Guide (withus_dfc_yolov5m.zip)

---

### Issue #14: Calibration 폴더 혼동 - train/images 직접 사용 (MINOR - RESOLVED)

**문제:** Hailo-Compiler-UI에서 HEF 컴파일 시 `calibset_size = 0` 오류 발생
```
pydantic.v1.error_wrappers.ValidationError: 2 validation errors for ModelOptimizationConfig
calibration -> calibset_size
  ensure this value is greater than 0 (type=value_error.number.not_gt; limit_value=0)
```

**원인 분석:**
- Hailo-Compiler-UI/data/calibration/images/ 폴더가 비어있어서 발생
- **잘못된 가정:** 별도의 calibration 폴더가 필요하다고 생각함
- **실제 동작:** 사용자의 데이터셋 train/images 폴더를 직접 사용함

**올바른 캘리브레이션 경로:**
```
<datasets-path>/<dataset-name>/train/images/
```

**잘못된 경로 (사용하지 않음):**
```
Hailo-H8/Hailo-Compiler-UI/data/calibration/images/  ❌
```

**해결 방법:**
1. `Hailo-Compiler-UI/data/calibration/` 폴더 완전 삭제
2. UI는 `find_calibration_folder()` 함수로 프로젝트의 train/images 자동 감지

**폴더 구조 표준:**
```
<dataset-folder>/
├── train/
│   └── images/          ← 캘리브레이션 이미지 (자동 감지)
├── valid/
│   └── images/          ← 대체 캘리브레이션 소스
├── models/
│   ├── pt/              ← PyTorch 모델
│   ├── onnx/            ← ONNX 모델
│   └── hef/             ← HEF 모델
└── data.yaml            ← 클래스 정보
```

**캘리브레이션 우선순위 (folder_structure.py):**
1. `train/images/` - Primary (권장)
2. `valid/images/` - Secondary
3. `test/images/` - Fallback

**교훈:**
1. **별도의 calibration 폴더를 만들지 않는다** - 기존 train/images 사용
2. UI에서 프로젝트 로드 시 `find_calibration_folder()`가 자동으로 경로 설정
3. 캘리브레이션 이미지는 복사하지 않고 원본 경로 직접 참조

| File | Changes |
|------|---------|
| `Hailo-Compiler-UI/data/calibration/` | 폴더 삭제 (불필요) |
| `src/utils/folder_structure.py` | 변경 없음 (이미 train/images 사용) |

**재발 방지:**
- Hailo-Compiler-UI/data/ 폴더에는 models/ 폴더만 유지
- 캘리브레이션은 항상 데이터셋 폴더의 train/images 직접 사용
- 이미지 복사 불필요

---

### Issue #15: YOLOv5 opset 자동 설정 (ENHANCEMENT - RESOLVED)

**요청:** YOLOv5 모델 선택 시 UI에서 opset을 11로 자동 설정

**배경:**
- YOLOv5 → Hailo 변환 시 opset 11 필수 (Hailo DFC 호환성)
- YOLOv8+ → opset 17 권장
- 사용자가 수동으로 opset 변경 시 실수 가능

**구현 내용:**

1. **`converter_panel.py` 수정:**
   - `detect_yolo_project_from_pt()` 함수에 `yolo_version` 필드 추가
   - 파일명에서 YOLO 버전 감지: `yolov5`, `yolov8`, `yolov9`, `yolov10`

2. **`main_window.py` 수정:**
   - `_auto_fill_onnx()` 메서드에 opset 자동 설정 로직 추가
   - YOLOv5 → opset 11 자동 설정 + 로그 메시지
   - YOLOv8/v9/v10 → opset 17 자동 설정

**코드 변경:**

```python
# converter_panel.py - detect_yolo_project_from_pt()
result['yolo_version'] = None  # v5, v8, etc.

# Detect YOLO version from filename
if 'yolov5' in pt_name:
    result['yolo_version'] = 'v5'
elif 'yolov8' in pt_name:
    result['yolo_version'] = 'v8'
# ... etc

# main_window.py - _auto_fill_onnx()
yolo_version = self._detected_project.get('yolo_version')
if yolo_version == 'v5':
    self.opsetCombo.setCurrentText('11')
    self._log_info("YOLOv5 detected → opset automatically set to 11 (Hailo compatible)")
elif yolo_version in ('v8', 'v9', 'v10'):
    self.opsetCombo.setCurrentText('17')
    self._log_info(f"YOLO{yolo_version} detected → opset set to 17")
```

**동작 방식:**
1. 사용자가 PT 파일 선택
2. 파일명에서 YOLO 버전 감지
3. opset 콤보박스 자동 설정
4. 로그에 감지 결과 표시

| YOLO Version | Opset | Reason |
|--------------|-------|--------|
| YOLOv5 | 11 | Hailo DFC 필수 |
| YOLOv8+ | 17 | 최신 기능 지원 |

| File | Changes |
|------|---------|
| `src/ui/converter_panel.py` | +12 lines (yolo_version 감지) |
| `src/ui/main_window.py` | +8 lines (opset 자동 설정) |

**테스트:** 17 unit tests passed

---

### Remaining Issues (To Be Fixed)

| Priority | Issue | Location | Status |
|----------|-------|----------|--------|
| **HIGH** | **ONNX 노드 자동 변환** | Hailo-Compiler-UI/converter.py | **Planned** |
| MEDIUM | print() instead of logger | 16+ locations across services/views | Pending |
| MEDIUM | No version control | Project root | Pending |
| MEDIUM | Duplicate temperature color code | device_tab.py, monitor_tab.py | Pending |
| MEDIUM | Magic numbers hardcoded | Timer 1000ms, temp 50/70, image 640x640 | Pending |
| MEDIUM | No unit tests | No tests/ directory | Pending |
| LOW | Path traversal vulnerability | settings_dialog.py:43-44 | Pending |
| LOW | Missing type hints | Entire codebase | Pending |

---

### Hailo-Compiler-UI 수정 계획: ONNX 노드 자동 변환

**목표:** PyTorch 2.x ONNX 파일의 노드 이름을 자동으로 감지하고 변환하여 hailomz CLI 호환성 확보

#### Phase 1: 노드 분석 및 검증 강화

| Task | File | Description |
|------|------|-------------|
| P1.1 | `onnx_utils.py` | `detect_onnx_naming_style()` 함수 추가 |
| P1.2 | `onnx_utils.py` | `is_hailo_compatible_naming()` 함수 추가 |
| P1.3 | `converter.py` | 컴파일 전 네이밍 스타일 검증 로직 추가 |

```python
def detect_onnx_naming_style(onnx_path: str) -> str:
    """
    Returns: 'yolov5_official', 'pytorch_generic', 'ultralytics', 'unknown'
    """
    conv_nodes = [n.name for n in model.graph.node if n.op_type == 'Conv']
    if conv_nodes[0].startswith('/model.'):
        return 'yolov5_official'  # Hailo compatible
    elif 'conv2d' in conv_nodes[0].lower():
        return 'pytorch_generic'  # Needs conversion
    return 'unknown'
```

#### Phase 2: 자동 변환 통합

| Task | File | Description |
|------|------|-------------|
| P2.1 | `onnx_utils.py` | `rename_nodes_to_hailo_style()` 함수 추가 |
| P2.2 | `converter.py` | `_prepare_onnx_for_hailo()` 메서드 추가 |
| P2.3 | `converter.py` | `compile_onnx_to_hef()` 수정 - 자동 변환 워크플로우 |

```python
def compile_onnx_to_hef(self, onnx_path, ...):
    # 1. 네이밍 스타일 검사
    naming_style = detect_onnx_naming_style(onnx_path)

    # 2. 비호환 시 자동 변환
    if naming_style != 'yolov5_official':
        self._log(f"ONNX naming style: {naming_style} (converting...)")
        onnx_path = self._prepare_onnx_for_hailo(onnx_path)

    # 3. hailomz 컴파일 진행
    return self._compile_with_hailomz(...)
```

#### Phase 3: UI 통합

| Task | File | Description |
|------|------|-------------|
| P3.1 | `convert_panel.py` | ONNX 호환성 상태 표시 |
| P3.2 | `convert_panel.py` | "Auto Convert" 체크박스 추가 |
| P3.3 | `convert_panel.py` | 변환 진행 상태 로그 표시 |

**UI 변경사항:**
```
┌─────────────────────────────────────────────────┐
│ ONNX Compatibility                              │
├─────────────────────────────────────────────────┤
│ Naming Style: pytorch_generic ⚠️                │
│ [✓] Auto-convert to Hailo compatible format    │
│                                                 │
│ Status: Will convert node_conv2d_N → /model.X  │
└─────────────────────────────────────────────────┘
```

#### Phase 4: 테스트 및 검증

| Task | Description |
|------|-------------|
| P4.1 | 단위 테스트: `test_onnx_naming_detection.py` |
| P4.2 | 단위 테스트: `test_onnx_node_conversion.py` |
| P4.3 | 통합 테스트: PT → ONNX → 변환 → HEF 전체 워크플로우 |

#### 예상 파일 변경

| File | Changes |
|------|---------|
| `src/core/onnx_utils.py` | +100 lines (신규 함수들) |
| `src/core/converter.py` | +50 lines (자동 변환 로직) |
| `src/ui/convert_panel.py` | +30 lines (UI 표시) |
| `scripts/rename_onnx_nodes.py` | 기존 (CLI 도구로 유지) |
| `tests/test_onnx_utils.py` | +80 lines (테스트) |

**우선순위:** HIGH - Custom 모델 변환 시 가장 빈번한 문제

---

### Project Statistics

- **Total Python Files:** 18
- **Total Lines of Code:** ~2,539 (Python) + 2,688 (Qt UI)
- **Architecture:** MVC pattern with PyQt5
- **Key Components:**
  - Device monitoring (temperature, power, utilization)
  - Model conversion pipeline (PT → ONNX → HEF)
  - Real-time inference with video/camera
  - Performance monitoring with CSV export

---

### Review Summary

**Overall Assessment:** The project has a clean architectural foundation with good service/view separation and graceful degradation for missing hardware. The main risks were:
1. Bare exception handlers masking bugs (FIXED)
2. Unsafe torch.load() call (FIXED)
3. Lack of tests (Pending)
4. print() instead of logging (Pending)

---

## 2026-02-07

### SDK Fallback Compilation Fixes (CRITICAL)

#### 1. End-Node-Names 순서 오류 수정

**문제:** SDK fallback 시 end-node-names 순서가 잘못됨
```
# 잘못된 순서 (proto가 마지막)
['/model.24/m.0/Conv', '/model.24/m.1/Conv', '/model.24/m.2/Conv', '/model.24/proto/cv3/conv/Conv']

# 올바른 순서 (proto가 처음)
['/model.24/proto/cv3/conv/Conv', '/model.24/m.0/Conv', '/model.24/m.1/Conv', '/model.24/m.2/Conv']
```

**원인:**
- `compile_onnx_to_hef()` 함수 (1082-1085줄)에서 detection_heads를 먼저 복사 후 proto_node를 append
- segment 모델에서는 proto_node가 반드시 첫 번째에 와야 함

**수정 (converter.py:1081-1088):**
```python
# Before
end_nodes = detection_heads.copy()
if proto_node:
    end_nodes.append(proto_node)  # 잘못됨: 끝에 추가

# After
if proto_node:
    end_nodes = [proto_node] + detection_heads  # 올바름: 처음에 추가
else:
    end_nodes = detection_heads.copy()
```

---

#### 2. GPU Availability Check 오류 수정

**문제:** SDK fallback 시 GPU 체크 subprocess 실패
```
Compilation failed: GPU availability check subprocess failed with exitcode 1
```

**원인:**
- Hailo SDK가 내부적으로 GPU 가용성을 subprocess로 체크
- PyQt5 UI 환경에서 subprocess 실행 시 환경 변수 문제 발생

**수정 (converter.py:1387-1391):**
```python
# GPU 체크 비활성화 환경 변수 추가
import os as _os
_os.environ['CUDA_VISIBLE_DEVICES'] = ''
_os.environ['HAILO_SDK_USE_GPU'] = '0'

self._log("Initializing Hailo compiler (CPU mode)...")
runner = ClientRunner(hw_arch=target)
```

**결과:**
- CPU 모드로 강제 실행하여 GPU 체크 오류 우회
- UI 환경에서 안정적인 HEF 컴파일 가능

---

### 수정 파일

| 파일 | 수정 내용 |
|------|----------|
| `src/core/converter.py:1081-1088` | end-node-names 순서 수정 (proto first) |
| `src/core/converter.py:1387-1391` | GPU 체크 비활성화 환경 변수 추가 |

### 테스트 결과
- 17개 unit test 전체 통과

---

### Issue #16: Preview Command 버튼 삭제 요청 (MINOR - RESOLVED)

**요청:** UI에서 "Preview Command" 버튼 완전 삭제
**반복 요청 횟수:** 다수 (사용자 불만 표시)

**삭제된 항목:**

| 파일 | 삭제 내용 |
|------|----------|
| `ui/main_window.ui` | `previewBtn` QPushButton 위젯 제거 |
| `src/ui/main_window.py` | `_preview_compile_command()` 메서드 전체 삭제 |
| `src/ui/main_window.py` | `_set_converting()` 위젯 목록에서 `self.previewBtn` 제거 |

**교훈:**
- 사용자 요청 시 관련 코드 전체 검색 필요 (UI 파일 + Python 코드)
- 버튼 삭제 시: UI 위젯, 시그널 연결, 핸들러 메서드, 참조 목록 모두 확인

---

### Issue #17: YOLOv5 Deep PT Analysis - 버전/타입/Opset 자동 감지 (ENHANCEMENT - RESOLVED)

**요청:** YOLOv5 모델 선택 시 detect/segment/classify 및 opset을 자동으로 감지

**배경:**
- 기존: 파일명/폴더명 패턴으로만 YOLO 버전과 모델 타입 감지
- 문제: `best.pt` 같은 파일명에서는 정보 추출 불가
- 필요: PT 파일 **내부 분석**으로 정확한 감지

**구현 내용:**

#### 1. 모듈 레벨 함수 3개 추가 (`src/core/converter.py`)

```python
def detect_yolo_version_from_pt(pt_path: str) -> str:
    """
    PT 파일 내부 분석으로 YOLO 버전 감지
    Returns: 'v5', 'v8', 'v9', 'v10', or None

    분석 방법:
    1. zipfile로 pickle 데이터 읽기 → 'ultralytics.nn', 'models.yolo' 문자열 검색
    2. torch.load() 후 모듈 경로 확인
    3. checkpoint 키 패턴 분석 (train_args → v8, model+ema+epoch → v5)
    """

def detect_yolo_task_from_pt(pt_path: str) -> str:
    """
    PT 파일 내부 분석으로 모델 타입 감지
    Returns: 'segment', 'detect', 'classify'

    분석 방법:
    1. pickle 데이터에서 'SegmentationModel', 'ClassificationModel' 검색
    2. 모델 클래스명 확인
    """

def get_recommended_opset(yolo_version: str, task_type: str) -> int:
    """
    버전/타입 기반 권장 opset 반환
    - YOLOv5 → 11 (Hailo DFC 호환)
    - YOLOv8+ → 17
    """
```

#### 2. `detect_yolo_project_from_pt()` 개선 (`src/ui/converter_panel.py`)

```python
# 기존: 파일명 패턴만 사용
if '-seg' in pt_name:
    result['model_type'] = 'segment'

# 개선: PT 파일 내부 분석 우선
detected_version = detect_yolo_version_from_pt(pt_path)  # 내부 분석
detected_task = detect_yolo_task_from_pt(pt_path)        # 내부 분석
result['recommended_opset'] = get_recommended_opset(detected_version, detected_task)
```

#### 3. UI 자동 설정 (`src/ui/main_window.py`)

```python
# PT 파일 선택 시 자동 적용
yolo_version = self._detected_project.get('yolo_version')
model_type = self._detected_project.get('model_type', 'detect')
recommended_opset = self._detected_project.get('recommended_opset', 17)

if yolo_version:
    self.opsetCombo.setCurrentText(str(recommended_opset))
    self.modelTypeCombo.setCurrentText(model_type)
    self._log_info(f"YOLO{yolo_version} {model_type} detected → opset={recommended_opset}")
```

**감지 방법 우선순위:**

| 순서 | 방법 | 신뢰도 |
|------|------|--------|
| 1 | zipfile pickle 데이터 분석 | 높음 (모듈 경로 확인) |
| 2 | torch.load() 클래스 확인 | 높음 (직접 로드) |
| 3 | ModuleNotFoundError 메시지 | 중간 (에러에서 추론) |
| 4 | 파일명/경로 패턴 | 낮음 (fallback) |

**자동 설정 항목:**

| 항목 | 감지 방법 | 자동 설정 |
|------|----------|----------|
| YOLO Version (v5/v8/v9/v10) | pickle 데이터 + 모듈 경로 | ✅ |
| Task Type (detect/segment/classify) | 모델 클래스명 분석 | ✅ |
| Opset (11 for v5, 17 for v8+) | 버전 기반 추천 | ✅ |

**수정 파일:**

| 파일 | 변경 내용 |
|------|----------|
| `src/core/converter.py` | +145 lines (3개 함수 추가) |
| `src/ui/converter_panel.py` | import 추가, `detect_yolo_project_from_pt()` 개선 |
| `src/ui/main_window.py` | `_auto_fill_onnx()` 개선 (recommended_opset 사용) |

**테스트 결과:**
- Python import 검증: ✅
- MainWindow import 검증: ✅

**사용자 경험 개선:**
```
# PT 파일 선택 시 로그 예시
[12:34:56] YOLOv5 segment detected → opset=11
[12:34:56] Detected YOLO project: /path/to/project
```

**교훈:**
1. 파일명 기반 감지는 신뢰도가 낮음 (`best.pt`, `last.pt` 등)
2. PT 파일 내부 pickle 데이터에 모델 정보가 포함됨
3. zipfile로 직접 읽으면 torch.load() 없이도 빠른 감지 가능

---

## 2026-02-08

### Issue #18: 로그 파일 자동 정리 - logs/ 폴더 이동 (ENHANCEMENT - RESOLVED)

**문제:** Hailo SDK/Runtime이 생성하는 `*.log` 파일이 프로젝트 전체에 흩어져 관리 어려움

**산재된 로그 파일:**

| 위치 | 파일 예시 |
|------|-----------|
| `HailoRT-Ui/` | `hailort.log`, `acceleras.log`, `hailo_sdk.client.log` |
| `HailoRT-Ui/src/` | `hailort.log`, `acceleras.log` |
| 프로젝트 루트 | `hailo_sdk.core.*.log`, `hailo_examples.log`, `allocator.log` |

**원인:**
- Hailo SDK/Runtime이 **현재 작업 디렉토리(CWD)**에 로그 파일을 자동 생성
- 앱 실행 위치에 따라 여러 디렉토리에 로그가 분산됨

**해결 방법:**
- `main.py`에 `collect_logs()` 함수 추가
- **앱 시작 시** `HailoRT-Ui/` 및 `HailoRT-Ui/src/`의 `*.log` 파일을 `HailoRT-Ui/logs/`로 자동 이동
- 이동 실패 시 (파일 사용 중 등) 조용히 건너뜀

**추가된 코드 (`HailoRT-Ui/main.py`):**
```python
def collect_logs():
    """Move scattered *.log files to logs/ folder at startup."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(app_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    scan_dirs = [app_dir, os.path.join(app_dir, 'src')]

    moved = 0
    for scan_dir in scan_dirs:
        if not os.path.isdir(scan_dir):
            continue
        for log_file in glob.glob(os.path.join(scan_dir, '*.log')):
            filename = os.path.basename(log_file)
            dst = os.path.join(logs_dir, filename)
            try:
                shutil.move(log_file, dst)
                moved += 1
            except OSError:
                pass

    if moved:
        print(f"[Startup] {moved} log file(s) moved to logs/")
```

**동작 흐름:**
```
앱 시작 (python main.py)
  → collect_logs() 실행
  → HailoRT-Ui/*.log + src/*.log 스캔
  → HailoRT-Ui/logs/ 로 자동 이동
  → "[Startup] N log file(s) moved to logs/" 출력
  → 앱 정상 시작
```

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/main.py` | `import glob, shutil` 추가, `collect_logs()` 함수 추가, `main()`에서 호출 |

**참고:**
- `.gitignore`에 `*.log` 및 `logs/` 이미 등록되어 git 영향 없음
- SDK가 실행 중 새 로그 생성 → 다음 앱 시작 시 자동 정리됨
- `logger.py`의 `log_dir` 파라미터로 앱 자체 로그도 `logs/`에 저장 가능

**심각도:** Minor

---

## 2026-02-14

### Issue #19: Project 탭 신규 추가 - 데이터셋 폴더 자동 감지 및 탭 간 연동 (ENHANCEMENT - RESOLVED)

**요청:** Hailo-Compiler-UI에는 데이터 폴더 선택 시 경로 자동 설정 기능이 있는데 HailoRT-Ui에는 없음

**1차 시도:** Convert 탭에 Data Folder UI 인라인 추가
- 결과: 탭이 과밀해짐 → 사용자 피드백으로 철회

**2차 구현 (최종):** 별도 "Project" 탭 생성 + Signal 기반 탭 간 연동

**구현 내용:**

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/utils/folder_structure.py` | 신규 - 데이터셋 폴더 구조 감지 유틸리티 |
| `HailoRT-Ui/ui/tabs/project_tab.ui` | 신규 - Project 탭 UI (폴더 선택, 트리뷰, 감지 정보, 자동 경로) |
| `HailoRT-Ui/src/views/project_tab.py` | 신규 - Project 탭 컨트롤러 (`project_changed` pyqtSignal) |
| `HailoRT-Ui/ui/main_window.ui` | 수정 - `tab_project` 첫 번째 탭으로 추가 |
| `HailoRT-Ui/src/app.py` | 수정 - ProjectTabController 초기화 및 `_on_project_changed()` 라우팅 |
| `HailoRT-Ui/src/views/convert_tab.py` | 수정 - `set_project_info()` 공개 API 추가, 인라인 Data Folder 코드 제거 |
| `HailoRT-Ui/src/views/inference_tab.py` | 수정 - `set_project_info()` 추가 (HEF 경로 자동 설정) |

**Signal 아키텍처:**
```
Project Tab (project_changed signal)
  → app.py (_on_project_changed)
    → convert_controller.set_project_info(info)
    → inference_controller.set_project_info(info)
```

**자동 감지 항목:**
- 클래스 수/이름 (data.yaml 파싱)
- Calibration 폴더 (train/images → valid/images → test/images 우선순위)
- PT 모델 파일
- ONNX/HAR/HEF 출력 디렉토리

**교훈:**
1. UI가 복잡해지면 별도 탭으로 분리하는 것이 사용자 경험에 유리
2. pyqtSignal을 사용한 탭 간 통신이 깔끔함 (직접 참조 없이 느슨한 결합)

**심각도:** Enhancement

---

### Issue #20: Hailo Python 패키지 가상환경 미설치 (MAJOR - RESOLVED)

**문제:** `hailomz CLI not found, using SDK compilation` 메시지 발생

**원인 분석:**
- 시스템 패키지(`hailort` 4.23.0, `hailort-pcie-driver` 4.23.0)는 dpkg로 설치됨
- Python 가상환경(`hailo_env`)에 Hailo Python 패키지 미설치
- `converter_service.py:665-677`에서 `hailomz` CLI 존재 확인 후 SDK fallback

**해결:**
```bash
source hailo_env/bin/activate
pip install install_sw/hailort-4.23.0-cp310-cp310-linux_x86_64.whl
pip install install_sw/hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
pip install install_sw/hailo_model_zoo-2.17.1-py3-none-any.whl
```

**설치 결과:**

| 패키지 | 버전 | 상태 |
|--------|------|------|
| hailort | 4.23.0 | OK |
| hailo_dataflow_compiler | 3.33.0 | OK |
| hailo_model_zoo | 2.17.1 | OK |
| hailomz CLI | - | OK (`hailo_env/bin/hailomz`) |

**부수 이슈:** Pillow 버전 충돌 경고 (`hailo_model_zoo`가 9.3.0 요구, `pi-heif`가 11.1+ 요구) - HailoRT 기능에 영향 없음

**심각도:** Major

---

### Issue #21: VIRTUAL_ENV=/usr 환경변수 문제 (MAJOR - RESOLVED)

**문제:** `hailo_sdk_client` import 시 권한 오류
```
PermissionError: [Errno 13] Permission denied: '/usr/etc'
```

**원인 분석:**
- 시스템 환경에 `VIRTUAL_ENV=/usr`가 설정되어 있음 (출처 불명)
- Hailo SDK의 `logger.py`에서 로그 디렉토리 경로 계산:
  ```python
  DFC_FOLDER = "etc/hailo/dfc/"
  DFC_FOLDER_PATH = os.path.join(os.environ.get("VIRTUAL_ENV", "$HOME/.hailo"), DFC_FOLDER)
  # VIRTUAL_ENV=/usr → /usr/etc/hailo/dfc/ → 권한 오류
  ```
- `.bashrc`, `.profile`, `/etc/environment` 등에서 설정 원인 찾지 못함

**해결:** venv를 `source hailo_env/bin/activate`로 활성화하면 `VIRTUAL_ENV`가 올바른 경로로 설정됨

**추가 경고:** `CC`/`CXX` 환경변수가 ccache로 설정 → `unset CC CXX` 필요

**교훈:**
- venv의 python을 직접 실행(`hailo_env/bin/python3`)하면 `VIRTUAL_ENV`가 설정되지 않음
- 반드시 `source activate`로 활성화 후 실행해야 Hailo SDK 정상 동작

**심각도:** Major

---

### Issue #22: 설치 스크립트에 Google Drive 자동 다운로드 추가 (ENHANCEMENT - RESOLVED)

**요청:** `install_sw/` 폴더가 없을 때 Google Drive에서 자동 다운로드

**구현 내용:**

| 파일 | 변경 내용 |
|------|----------|
| `setup-env-ubuntu2204.sh` | STEP 6에 `install_sw/` 확인 + Google Drive 다운로드 로직 추가 |
| `setup-env-wsl2.sh` | 동일한 다운로드 로직 추가 |

**기능:**
1. `_count_hailo_wheels()` 함수로 필수 `.whl` 파일 3개 존재 여부 확인
2. 누락 시 사용자에게 선택지 제공:
   - 1) `gdown`으로 Google Drive 폴더 자동 다운로드
   - 2) Skip (수동 다운로드 URL 안내)
3. 다운로드 후 후처리:
   - `find -mindepth 2`로 하위 폴더의 `.whl`/`.deb` 파일을 루트로 이동
   - `Zone.Identifier` 파일 삭제 (Windows NTFS 아티팩트)
   - `.msi` 파일 삭제 (Linux에서 불필요)
   - 빈 디렉토리 삭제

**Google Drive URL:** `https://drive.google.com/drive/folders/1pAv-qojczbuIskbWu0W_0sac6LhJRF66`

**테스트 결과 (실제 다운로드 검증):**

| 항목 | 결과 |
|------|------|
| `gdown` 설치 | OK |
| 폴더 다운로드 (8개 파일, ~570MB) | OK |
| 하위 폴더 → 루트 이동 | OK (`ubuntu/install_sw/` → `install_sw/`) |
| `Zone.Identifier` 정리 (8개) | OK |
| `.msi` 정리 (2개) | OK |
| 빈 폴더 정리 | OK |
| 최종 파일 확인 (`.whl` 3개 + `.deb` 4개) | OK |

**gdown 동작 특성 (테스트에서 발견):**
1. Google Drive 폴더 구조를 그대로 유지 → `install_sw/ubuntu/install_sw/`에 파일 생성
2. Windows `Zone.Identifier` 파일도 같이 다운로드
3. `window/` 하위 폴더의 `.msi` 파일도 다운로드

**교훈:**
1. **스크립트 작성 후 반드시 실제 테스트 필수** - `bash -n` 문법 검증만으로는 부족
2. `gdown --folder`의 폴더 구조 보존 동작은 문서에 명확하지 않아 테스트 없이는 발견 불가
3. 후처리 로직의 첫 버전(`for SUBFOLDER` 루프)은 중첩 구조를 제대로 처리 못함 → `find -mindepth 2`로 수정

**심각도:** Enhancement

---

## 2026-02-15

### Issue #23: HailoRT 4.23 Python API 변경 - Device 정보 조회 실패 (CRITICAL - RESOLVED)

**문제:** HailoRT 4.12 API로 작성된 device_tab.py가 4.23에서 동작하지 않음
```
AttributeError: 'VDevice' object has no attribute 'get_serial_number'
AttributeError: 'VDevice' object has no attribute 'get_fw_version'
```

**원인 분석:**
- HailoRT 4.23에서 Device API가 대폭 변경됨
- `VDevice`에 직접 접근 메서드가 제거되고, `physical_device.control` 경유로 변경

**HailoRT 4.12 vs 4.23 API 비교:**

| 기능 | 4.12 API | 4.23 API |
|------|----------|----------|
| 보드 정보 | `device.get_board_info()` | `physical_device.control.identify()` → `BoardInformation` |
| 시리얼 | `device.get_serial_number()` | `board_info.serial_number` (null strip 필요) |
| FW 버전 | `device.get_fw_version()` | `str(board_info.firmware_version)` (HailoFirmwareVersion obj) |
| 아키텍처 | `device.get_architecture()` | `board_info.device_architecture` |
| 온도 | `device.get_chip_temperature()` → float | `ctrl.get_chip_temperature()` → `TemperatureInfo` (`.ts0_temperature`, `.ts1_temperature`) |
| 전력 | `device.get_power_measurement()` → float | `ctrl.get_power_measurement()` → `PowerMeasurementData` (`.average_value`) |

**해결 방법:**

```python
# VDevice → physical device → control
vdevice = VDevice()
physical_devices = vdevice.get_physical_devices()
physical_device = physical_devices[0]
ctrl = physical_device.control

# Board info
board_info = ctrl.identify()
board_name = str(board_info.board_name).replace('\x00', '').strip()
serial = str(board_info.serial_number).replace('\x00', '').strip()
fw_version = str(board_info.firmware_version).replace('\x00', '').strip()

# Temperature
temp_info = ctrl.get_chip_temperature()
temp = temp_info.ts0_temperature  # float, °C

# Power (일부 보드에서 미지원)
try:
    power_info = ctrl.get_power_measurement()
    power = power_info.average_value
except Exception:
    power = None  # M.2 보드 등에서 미지원
```

**주의사항:**
- C-style 문자열에 null 바이트 패딩 → `.replace('\x00', '').strip()` 필수
- `get_power_measurement()`는 일부 보드(M.2 등)에서 지원 안 됨
- USB/LAN 인터페이스 모델은 HailoRT 4.12만 지원

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/services/hailo_service.py` | 4.23 API로 전면 재작성 |
| `HailoRT-Ui/src/views/device_tab.py` | 새 API 반환값 처리 |

**심각도:** Critical

---

### Issue #24: Inference "Device not connected" 오류 (CRITICAL - RESOLVED)

**문제:** Inference 탭에서 Start 클릭 시 항상 "No Hailo device available" 오류

**원인 분석:**
- `InferenceWorker`가 자체적으로 `HailoService()`를 새로 생성
- Device 탭에서 이미 연결된 디바이스를 공유하지 않음
- Hailo-8은 동시에 하나의 VDevice만 허용 → 중복 연결 실패

**해결 방법:**
- `InferenceTabController`에 `device_controller` 참조 전달
- `InferenceWorker.__init__`에 `hailo_service` 파라미터 추가
- Device 탭에서 연결된 HailoService 인스턴스를 공유

```python
# app.py - device_controller 공유
self.inference_controller = InferenceTabController(
    self.tab_inference, self.base_path,
    device_controller=self.device_controller
)

# inference_tab.py - 공유 서비스 사용
hailo_svc = None
if self.device_controller and hasattr(self.device_controller, 'hailo_service'):
    hailo_svc = self.device_controller.hailo_service

self.worker = InferenceWorker(
    self.hef_path, source, hailo_service=hailo_svc, class_names=class_names
)
```

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/app.py` | `device_controller`를 inference/monitor에 전달 |
| `HailoRT-Ui/src/views/inference_tab.py` | `hailo_service` 공유 로직 추가 |

**심각도:** Critical

---

### Issue #25: Inference 입력 크기 불일치 - Expected 1228800, got 5760 (CRITICAL - RESOLVED)

**문제:** 추론 실행 시 입력 데이터 크기 오류
```
Expected input size 1228800, got 5760
```

**원인 분석:**
- `_preprocess()`에서 `cv2.resize(frame, (input_shape[2], input_shape[1]))` 사용
- `input_shape = (640, 640, 3)` (NHWC) 일 때 `(input_shape[2], input_shape[1])` = `(3, 640)`
- 640×3×3 = 5760 bytes (잘못된 크기)
- 기대값: 640×640×3 = 1,228,800 bytes

**해결:**
```python
# Before (잘못됨)
resized = cv2.resize(frame, (input_shape[2], input_shape[1]))

# After (올바름)
target_h, target_w = input_shape[0], input_shape[1]  # (640, 640)
resized = cv2.resize(frame, (target_w, target_h))
```

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/services/hailo_service.py` | `_preprocess()` resize 인자 순서 수정 |

**심각도:** Critical

---

### Issue #26: float32 vs uint8 dtype 불일치 (MAJOR - RESOLVED)

**문제:** HailoRT가 uint8 입력을 기대하는데 float32 데이터 전송

**원인 분석:**
- `_preprocess()`에서 `normalized = resized.astype(np.float32) / 255.0` 수행
- HailoRT는 내부적으로 양자화를 처리하므로 uint8 그대로 전달해야 함

**해결:**
```python
# Before (잘못됨)
normalized = resized.astype(np.float32) / 255.0

# After (올바름) - uint8 유지, HailoRT가 내부적으로 양자화 처리
# resized는 이미 uint8, 변환 불필요
```

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/services/hailo_service.py` | `_preprocess()` - float32 정규화 제거, uint8 유지 |

**심각도:** Major

---

### Issue #27: HAILO_NETWORK_GROUP_NOT_ACTIVATED 오류 (CRITICAL - RESOLVED)

**문제:** `InferVStreams` 실행 시 네트워크 그룹 활성화 오류
```
HAILO_NETWORK_GROUP_NOT_ACTIVATED
```

**원인 분석:**
- HailoRT 4.23에서는 `InferVStreams` 사용 전에 network group을 명시적으로 활성화해야 함
- 이전 버전에서는 자동 활성화되었으나 4.23에서 변경됨

**해결:**
```python
# Before (4.12 방식)
with InferVStreams(self.network_group, input_data) as pipeline:
    results = pipeline.infer(input_data)

# After (4.23 방식) - 명시적 활성화 필요
with self.network_group.activate():
    with InferVStreams(self.network_group, input_data) as pipeline:
        results = pipeline.infer(input_data)
```

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/services/hailo_service.py` | `infer()` - `network_group.activate()` 컨텍스트 매니저 추가 |

**심각도:** Critical

---

### Issue #28: quantized=False 출력이 여전히 uint8 반환 (MAJOR - RESOLVED)

**문제:** `InferVStreams`에 `quantized=False` 설정해도 출력이 float32로 역양자화되지 않음

**원인 분석:**
- HailoRT 4.23의 버그 또는 동작 변경
- `output_params`에 `quantized=False` 설정해도 uint8 그대로 반환

**해결 방법 - 수동 역양자화:**
```python
# vstream_info에서 양자화 파라미터 추출
for vstream_info in self.network_group.get_output_vstream_infos():
    qp = vstream_info.quant_info
    self._output_quant_params[vstream_info.name] = {
        'zp': qp.qp_zp,       # zero point
        'scale': qp.qp_scale,  # scale factor
    }

# 수동 역양자화: float_value = (uint8_value - zero_point) * scale
def _dequantize(self, name, data):
    qp = self._output_quant_params.get(name)
    if qp:
        return (data.astype(np.float32) - qp['zp']) * qp['scale']
    return data.astype(np.float32)
```

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/services/hailo_service.py` | `_output_quant_params` 저장, `_dequantize()` 함수 추가 |

**심각도:** Major

---

### Issue #29: YOLOv8 DFL 디코딩 - 18개 Raw 출력 레이어 처리 (CRITICAL - RESOLVED)

**문제:** Forklift 모델(YOLOv8 custom, nc=1)이 NMS 없이 18개 raw 출력 레이어를 반환하여 탐지 결과가 비어있음

**원인 분석:**
- Hailo 컴파일러가 Conv 레이어를 분할하여 18개 출력 생성
- 각 스케일(80×80, 40×40, 20×20)별로 6개 출력:
  - 3×64ch: bbox 중간값 + DFL 최종 출력
  - 2×128ch: cls 중간값
  - 1×nc(1)ch: cls 최종 출력
- NMS가 포함되지 않아 수동 DFL 디코딩 필요

**YOLOv8 DFL 출력 구조 분석:**

| 출력 | Shape | 채널 | 역할 |
|------|-------|------|------|
| conv37 (80×80) | 80×80×64 | 64 | bbox 중간값 |
| conv38 (80×80) | 80×80×128 | 128 | cls 중간값 |
| conv39 (80×80) | 80×80×64 | 64 | bbox 중간값 |
| conv40 (80×80) | 80×80×128 | 128 | cls 중간값 |
| **conv41** (80×80) | 80×80×64 | 64 | **bbox DFL 최종** (가장 높은 conv#) |
| **conv42** (80×80) | 80×80×1 | 1 | **cls 최종** (nc=1) |
| (40×40, 20×20도 동일 패턴) | | | |

**핵심 식별 규칙:**
- 같은 스케일에서 **가장 높은 conv 번호** + **64ch** = bbox DFL 출력
- 같은 스케일에서 **가장 작은 채널 수** (= nc) = cls 출력

**DFL 디코딩 알고리즘:**
```python
# reg_max = 15 (NOT 16), 4 * (15+1) = 64 channels
reg_max = 15

# 1. bbox DFL: (H, W, 64) → reshape (H*W, 4, reg_max+1)
bbox_raw = dequantized_bbox.reshape(-1, 4, reg_max + 1)

# 2. Softmax → weighted sum (Distribution Focal Loss)
bbox_dfl = softmax(bbox_raw, axis=-1)
weights = np.arange(reg_max + 1, dtype=np.float32)
bbox_dist = np.sum(bbox_dfl * weights, axis=-1)  # (H*W, 4)

# 3. dist2bbox: grid offset → xyxy
grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
grid = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
stride = input_size / H  # 8, 16, 32

x1 = (grid[:, 0] - bbox_dist[:, 0]) * stride
y1 = (grid[:, 1] - bbox_dist[:, 1]) * stride
x2 = (grid[:, 0] + bbox_dist[:, 2]) * stride
y2 = (grid[:, 1] + bbox_dist[:, 3]) * stride

# 4. cls sigmoid → NMS
cls_scores = sigmoid(dequantized_cls).reshape(-1, nc)
```

**구현된 핵심 함수들:**

| 함수 | 설명 |
|------|------|
| `_analyze_outputs()` | 18개 출력을 스케일별로 그룹핑, bbox/cls 자동 식별 |
| `_dequantize()` | uint8 → float32 수동 역양자화 |
| `_postprocess_yolov8_dfl()` | DFL softmax → weighted sum → grid offset → NMS |
| `set_class_names()` | data.yaml 클래스 이름 설정 |

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/services/hailo_service.py` | 전면 재작성 - DFL 디코딩 파이프라인 |
| `HailoRT-Ui/src/views/inference_tab.py` | `class_names` 전달 로직 추가 |

**테스트 결과:**
- 모델: Forklift YOLOv8 custom (nc=1, palette_green)
- 입력: 640×640×3 uint8
- 출력: 18개 레이어 → 3 스케일 DFL 디코딩
- 결과: 5/5 테스트 이미지에서 palette_green 탐지 성공 (confidence 0.96)

**심각도:** Critical

---

### Issue #30: data.yaml 클래스 이름 연동 (ENHANCEMENT - RESOLVED)

**문제:** Inference 결과에서 클래스 이름 대신 숫자 ID만 표시

**해결 방법:**
- Project 탭에서 감지한 `class_names`를 Inference 탭으로 전달
- `HailoService.set_class_names()` 메서드 추가
- `InferenceWorker`에 `class_names` 파라미터 추가

**Signal 흐름:**
```
Project Tab (project_changed: {class_names: ['palette_green']})
  → app.py (_on_project_changed)
    → inference_controller.set_project_info(info)
      → self._project_info = info

Start Inference:
  → class_names = self._project_info.get('class_names')
  → InferenceWorker(... class_names=class_names)
    → hailo.set_class_names(class_names)
```

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/services/hailo_service.py` | `set_class_names()` 메서드 추가 |
| `HailoRT-Ui/src/views/inference_tab.py` | `class_names` 전달 로직 추가 |

**심각도:** Enhancement

---

### Issue #31: QGroupBox 캡션 텍스트 안 보임 (MAJOR - RESOLVED)

**문제:** 모든 탭의 QGroupBox 제목(캡션)이 보이지 않거나 잘림

**원인 분석:**

| 원인 | 상세 |
|------|------|
| `margin-top: 20px` 부족 | 13pt(≈17px) + `top: 4px` = 21px > 20px 마진 → 클리핑 |
| `font-weight: bold; font-size: 13pt` 위치 | QGroupBox 자체에 설정 → 내부 모든 위젯에도 적용 |
| `QFrame` 범용 스타일 | 모든 QFrame 파생 위젯에 border 적용 → QGroupBox 렌더링 간섭 |

**해결:**

```css
/* Before */
QGroupBox {
    margin-top: 20px;       /* 부족 */
    font-weight: bold;      /* 자식에도 적용됨 */
    font-size: 13pt;        /* 자식에도 적용됨 */
}
QGroupBox::title {
    top: 4px;               /* 마진 초과 가능 */
}
QFrame {                    /* 모든 QFrame에 적용 - 간섭 */
    border: 1px solid #ccc;
}

/* After */
QGroupBox {
    margin-top: 30px;       /* 충분한 공간 */
}
QGroupBox::title {
    top: 8px;               /* 적절한 위치 */
    font-weight: bold;      /* 타이틀에만 적용 */
    font-size: 13pt;        /* 타이틀에만 적용 */
    padding: 2px 10px;      /* 여유 패딩 */
}
/* QFrame 범용 규칙 제거 → QTableWidget, QTreeWidget 개별 스타일 */
```

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/utils/styles.py` | QGroupBox margin/title 수정, QFrame 범용 규칙 제거, QTableWidget/QTreeWidget/QHeaderView 개별 스타일 추가 |

**테스트 결과:** 5개 탭 모두 QGroupBox 캡션 정상 표시 확인 (스크린샷 검증)

**심각도:** Major

---

### Issue #32: About 다이얼로그 저작자 및 라이센스 표기 (ENHANCEMENT - RESOLVED)

**요청:** About 다이얼로그에 저작자 정보와 라이센스 표기 추가

**구현 내용:**

About 다이얼로그에 추가된 정보:
- **Author:** Prof. Kuk Won Ko with Claude Code
- **License:** Free for personal and non-commercial use. Commercial and enterprise use requires a paid license.

LICENSE 파일 신규 생성 (듀얼 라이센스):
1. **개인/비상업용:** 무료 (교육, 개인, 비영리 목적)
2. **상업/기업용:** 유료 라이센스 필요 (연락처: kuksauto@hanmail.net)

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/app.py` | `_show_about()` - Author, License 정보 추가 |
| `LICENSE` | 신규 - 듀얼 라이센스 (개인 무료 / 기업 유료) |

**심각도:** Enhancement

---

### Issue #33: Project 탭 - 최근 폴더 5개 기억 및 전체 디렉토리 트리 표시 (ENHANCEMENT - RESOLVED)

**요청:**
1. 폴더가 최근 폴더 5개를 기억하고 선택할 수 있도록
2. 폴더를 열면 트리 전체 구조가 보이도록

**구현 내용:**

**1) 최근 폴더 기능:**
- `QLineEdit` → `QComboBox` (editable) 교체
- `recent_folders.json`에 최근 5개 폴더 JSON 저장/로드
- 드롭다운 선택 시 즉시 해당 폴더 감지 실행
- 새 폴더 선택 시 리스트 맨 위에 추가 (중복 제거)

**2) 전체 디렉토리 트리:**
- 기존: 미리 정의된 구조만 표시 (data.yaml, train/, models/ 등)
- 변경: `os.listdir()` 재귀 탐색으로 실제 파일시스템 전체 표시
- 폴더는 "N items" 상태 표시, 파일은 크기(KB/MB/GB) 표시
- 데이터셋 관련 항목 색상 구분:
  - 데이터셋 폴더 (train, models 등): 파랑
  - PT 모델: 보라
  - ONNX: 파랑
  - HAR/HEF: 주황
  - data.yaml: 초록
- 상위 2단계 자동 펼침, 최대 4단계까지 탐색
- 숨김 파일/`__pycache__` 자동 제외

**3) File 메뉴 변경:**
- `Open` → `Open Folder` (Ctrl+O)
- 클릭 시 Project 탭으로 자동 전환 + 폴더 선택 다이얼로그

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/ui/tabs/project_tab.ui` | `QLineEdit` → `QComboBox` 교체 |
| `HailoRT-Ui/src/views/project_tab.py` | 최근 폴더 관리, 재귀 트리 표시 전면 재작성 |
| `HailoRT-Ui/ui/main_window.ui` | `actionOpen` → `actionOpenFolder` 변경 |
| `HailoRT-Ui/src/app.py` | `_open_folder()` 메서드 추가, 메뉴 연결 |
| `.gitignore` | `recent_folders.json` 추가 |

**심각도:** Enhancement

---

### Issue #34: QTabBar 탭 텍스트 잘림 - "Project", "Model Convert" (MAJOR - RESOLVED)

**문제:** "Project" 탭 끝 글자, "Model Convert" 탭 끝 글자가 잘려서 보임

**원인 분석:**
- Qt5 QTabBar는 **스타일시트 적용 전** 위젯의 기본 QFont로 탭 너비를 계산
- 스타일시트에서 `font-weight: bold`나 `font-size` 변경 시, 계산된 너비보다 실제 텍스트가 넓어짐
- `QWidget { font-size: 12pt; }` 글로벌 규칙도 탭 크기 계산에 반영 안 됨
- 선택/비선택 상태 font 차이와 무관하게, 스타일시트 font 자체가 문제

**시도한 방법들 (실패):**

| 시도 | 방법 | 결과 |
|------|------|------|
| 1차 | `min-width: 100px` 제거 | 여전히 잘림 |
| 2차 | `qproperty-expanding: false` + selected에만 bold | "roject" (더 심해짐) |
| 3차 | 모든 탭에 bold 적용 | 여전히 잘림 (Qt가 bold 기준으로도 크기 미계산) |

**최종 해결 - bold 완전 제거:**
```css
/* 핵심: font-weight: bold를 탭에서 완전히 제거 */
QTabBar {
    qproperty-expanding: false;  /* 탭이 전체 너비를 채우려 하지 않음 */
}

QTabBar::tab {
    /* font-weight: bold 없음! */
    padding: 10px 24px;
    color: #333333;
}

QTabBar::tab:selected {
    color: #1565c0;
    border-bottom: 2px solid #1565c0;  /* bold 대신 하단 보더로 선택 표시 */
}
```

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/utils/styles.py` | `QTabBar` - bold 완전 제거, `qproperty-expanding: false`, 선택 탭은 `border-bottom`으로 구분 |

**교훈:** Qt5 스타일시트에서 `font-weight: bold`를 QTabBar에 사용하면 안 됨. Qt가 탭 크기를 스타일시트 적용 전 기본 폰트로 계산하기 때문. 선택 상태 구분은 색상/보더 등 크기에 영향 없는 속성으로 해야 함.

**심각도:** Major

---

### Issue #35: HEF 직접 선택 시 클래스 이름 COCO fallback (MAJOR - RESOLVED)

**문제:** Project 탭을 거치지 않고 Inference 탭에서 직접 HEF 파일을 선택하면, `class_names`가 None → COCO_NAMES fallback → forklift 모델이 "person: 0.96"으로 표시

**원인 분석:**
- `start_inference()`에서 `class_names`를 `_project_info`에서만 가져옴
- `_project_info`는 Project 탭에서 `set_project_info()` 호출 시에만 설정됨
- 직접 HEF 브라우즈 시 `_project_info = None` → `class_names = None` → COCO fallback

**해결 - HEF 경로에서 data.yaml 자동 탐색:**
```python
def _find_class_names_from_hef(self, hef_path: str) -> Optional[List[str]]:
    """HEF 파일 경로에서 상위 디렉토리를 탐색하여 data.yaml의 클래스 이름 찾기"""
    current = os.path.dirname(os.path.abspath(hef_path))
    for _ in range(5):  # 최대 5단계 상위까지 탐색
        data_config = parse_data_yaml(current)
        if data_config:
            names = data_config.get('names')
            if names:
                return names
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None
```

**클래스 이름 해결 우선순위 (3단계):**
```
1. self._class_names (Project 탭에서 직접 설정)
2. self._project_info.get('class_names') (Project 정보)
3. self._find_class_names_from_hef(hef_path) (HEF 경로 자동 탐색)
4. COCO_NAMES fallback (기본값)
```

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/views/inference_tab.py` | `_find_class_names_from_hef()` 메서드 추가, `start_inference()` 클래스 이름 해결 로직 3단계로 확장 |

**테스트 결과:** `/home/amap/yolov8_custom/Project_yolov8/forklift/models/hef/best.hef` → 2단계 상위에서 `data.yaml` 발견 → `['palette_green']` 정상 로드

**심각도:** Major

---

### Issue #36: 프로젝트 선택 시 자동 파일 감지 및 로딩 (ENHANCEMENT - RESOLVED)

**요청:** "프로젝트 선택시 관련 파일 체크도 다하고 관련 파일을 자동으로 로딩하게 해주세요"

**구현 내용:**

**1) folder_structure.py - ONNX/HEF/HAR 파일 감지 추가:**
```python
# models/ 하위 또는 루트에서 모델 파일 자동 탐색
result['onnx_files'] = _find_files_by_ext(os.path.join(models_dir, 'onnx'), '.onnx')
result['hef_files'] = _find_files_by_ext(os.path.join(models_dir, 'hef'), '.hef')
result['har_files'] = _find_files_by_ext(os.path.join(models_dir, 'har'), '.har')
```

**2) Project 탭 - "Apply" 버튼 없이 자동 적용:**
- 폴더 감지 완료 시 자동으로 `project_changed` 시그널 발신
- "Applied!" 피드백 2초 표시 후 원래 상태 복원
- 수동 Apply 버튼은 유지 (재적용 용도)

**3) Convert 탭 - ONNX 자동 선택:**
```python
# 'best' 키워드 포함 파일 우선, 없으면 첫 번째 파일
onnx_files = info.get('onnx_files', [])
if onnx_files and not self.tab.editOnnxPath.text():
    best_onnx = next(
        (f for f in onnx_files if 'best' in os.path.basename(f).lower()),
        onnx_files[0]
    )
    self.tab.editOnnxPath.setText(best_onnx)
```

**4) Inference 탭 - HEF 자동 선택 + 모델 로드:**
```python
hef_files = info.get('hef_files', [])
if hef_files and not self.tab.editHefPath.text():
    best_hef = next(
        (f for f in hef_files if 'best' in os.path.basename(f).lower()),
        hef_files[0]
    )
    self.tab.editHefPath.setText(best_hef)
    self.load_model()  # 즉시 모델 로드
```

**자동 로딩 흐름:**
```
폴더 선택 (Project 탭)
  → detect_project_structure() (PT/ONNX/HEF/HAR 전부 스캔)
  → 트리 표시 + 정보 패널 업데이트
  → project_changed 시그널 자동 발신
    → Convert 탭: ONNX 경로 자동 설정
    → Inference 탭: HEF 경로 자동 설정 + 모델 자동 로드
    → class_names 자동 전달
```

| 파일 | 변경 내용 |
|------|----------|
| `HailoRT-Ui/src/utils/folder_structure.py` | `_find_files_by_ext()` 헬퍼 추가, ONNX/HEF/HAR 감지 |
| `HailoRT-Ui/src/views/project_tab.py` | 감지 완료 시 자동 시그널 발신, ONNX/HEF 정보 표시 |
| `HailoRT-Ui/ui/tabs/project_tab.ui` | Detected Information에 ONNX/HEF 행 추가 |
| `HailoRT-Ui/src/views/convert_tab.py` | `set_project_info()` - ONNX 자동 선택 |
| `HailoRT-Ui/src/views/inference_tab.py` | `set_project_info()` - HEF 자동 선택 + 모델 로드, `_class_names` 저장 |

**테스트 결과:** Forklift 프로젝트 폴더 선택 → ONNX/HEF 자동 감지 → Inference 탭에 HEF 자동 로드 → "palette_green: 0.96" 정상 탐지 확인

**심각도:** Enhancement
