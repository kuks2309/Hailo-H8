# YOLOv5 → Hailo H8 HEF 변환 가이드

> **테스트 완료:** 2026-02-06 | **환경:** WSL2 Ubuntu 22.04, Python 3.10, HailoRT 4.23.0

## 개요

| 항목 | 값 |
|------|-----|
| 모델 | YOLOv5s-seg |
| 입력 크기 | 640x640 |
| 타겟 디바이스 | Hailo H8 |
| 예상 성능 | **92.34 FPS, 18.40 ms latency** |

---

## 환경 설정

### 필수 패키지

```bash
# Hailo 가상환경 활성화
source /path/to/hailo_env/bin/activate

# 확인
pip list | grep -E "hailo|torch|onnx"
```

**검증된 버전:**

| 패키지 | 버전 | 비고 |
|--------|------|------|
| hailo-dataflow-compiler | 3.33.0 | |
| hailo-model-zoo | 2.17.1 | |
| hailort | 4.23.0 | |
| onnx | 1.16.0 | **중요: 1.17+ 사용 금지** |
| torch | 2.10.0 | |
| ultralytics | 8.4.11 | |

---

## 방법 1: Hailo Model Zoo 표준 모델 (권장)

가장 안정적인 방법. Hailo가 검증한 ONNX 모델을 자동 다운로드하여 컴파일.

### Step 1: 캘리브레이션 이미지 준비

최소 64개 이상 권장 (테스트: 102개 사용)

```bash
# COCO128 샘플 다운로드
python3 -c "
import urllib.request, zipfile, shutil, glob
urllib.request.urlretrieve('https://ultralytics.com/assets/coco128.zip', 'coco128.zip')
with zipfile.ZipFile('coco128.zip', 'r') as z:
    z.extractall('.')
# 이미지를 calibration 폴더로 복사
for img in glob.glob('coco128/images/train2017/*.jpg')[:100]:
    shutil.copy(img, 'calibration_images/')
"
```

### Step 2: HEF 컴파일

```bash
hailomz compile yolov5s_seg \
    --hw-arch hailo8 \
    --calib-path /path/to/calibration_images \
    --classes 80
```

**컴파일 결과:**

| 항목 | 값 |
|------|-----|
| Mapping 시간 | ~4분 |
| Compilation 시간 | ~16초 |
| HEF 파일 크기 | 7.4 MB |
| HAR 파일 크기 | 150 MB |

### Step 3: 성능 검증

```bash
# 프로파일러 (디바이스 없이)
hailomz profile yolov5s_seg --har yolov5s_seg.har --hw-arch hailo8

# 벤치마크 (디바이스 필요)
hailo benchmark yolov5s_seg.hef
```

---

## 방법 2: 커스텀 모델 변환

자체 학습한 YOLOv5 모델을 변환할 때 사용.

### Step 1: ONNX Export (YOLOv5 공식 방법)

**중요:** YOLOv5 공식 `export.py` 사용 필수

```bash
cd /path/to/yolov5

python3 export.py \
    --weights best.pt \
    --include onnx \
    --imgsz 640 \
    --opset 11 \
    --simplify
```

**옵션 설명:**

| 옵션 | 값 | 이유 |
|------|-----|------|
| `--opset` | 11 | Hailo DFC 호환성 |
| `--simplify` | 필수 | 불필요한 노드 제거 |
| `--dynamic` | 사용 안 함 | 고정 입력 크기 필요 |

### Step 2: ONNX 노드 이름 확인 (필수)

```python
import onnx

model = onnx.load("best.onnx")

print("=== Output Names ===")
for output in model.graph.output:
    print(f"  {output.name}")

print("\n=== Last Conv Nodes ===")
conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
for node in conv_nodes[-5:]:
    print(f"  {node.name} -> {node.output[0]}")
```

**예상 출력 (Hailo 호환):**

```
=== Last Conv Nodes ===
  /model.24/proto/cv3/conv/Conv
  /model.24/m.0/Conv
  /model.24/m.1/Conv
  /model.24/m.2/Conv
```

### Step 3: HEF 컴파일

```bash
hailomz compile yolov5s_seg \
    --ckpt /path/to/best.onnx \
    --hw-arch hailo8 \
    --calib-path /path/to/train/images \
    --classes <num_classes> \
    --end-node-names /model.24/m.0/Conv /model.24/m.1/Conv /model.24/m.2/Conv /model.24/proto/cv3/conv/Conv
```

---

## 트러블슈팅

### 1. "Unable to find end node names" 에러

**원인:** PyTorch 2.x로 export된 ONNX의 노드 이름이 Hailo와 호환되지 않음

| Export 방식 | 노드 이름 패턴 | Hailo 호환 |
|-------------|---------------|-----------|
| YOLOv5 공식 export.py | `/model.24/m.X/Conv` | ✅ |
| PyTorch 2.x torch.onnx.export | `conv2d_N` | ❌ |

**해결:**
1. YOLOv5 공식 `export.py` 사용
2. 또는 `--end-node-names`에 실제 노드 이름 지정

### 2. ONNX opset 버전 에러

```
RuntimeError: No Adapter To Version $17 for Resize
```

**원인:** PyTorch 2.x는 내부적으로 opset 18 사용, 다운컨버트 실패

**해결:**
- YOLOv5 공식 export.py는 `--opset 11` 정상 작동
- 또는 Hailo Model Zoo 표준 모델 사용

### 3. `onnx.mapping` 모듈 에러

```
ModuleNotFoundError: No module named 'onnx.mapping'
```

**원인:** onnx 1.17+ 버전 사용

**해결:**
```bash
pip install onnx==1.16.0
```

### 4. 캘리브레이션 이미지 부족 경고

**원인:** 1024개 미만의 이미지

**해결:** 운영 환경에서는 더 많은 이미지 권장, 테스트용으로는 64개 이상이면 가능

---

## 지원 모델

| 모델 | hailomz 이름 | 비고 |
|------|-------------|------|
| YOLOv5n-seg | yolov5n_seg | |
| YOLOv5s-seg | yolov5s_seg | 테스트 완료 |
| YOLOv5m-seg | yolov5m_seg | |
| YOLOv5l-seg | yolov5l_seg | |
| YOLOv8n-seg | yolov8n_seg | |
| YOLOv8s-seg | yolov8s_seg | |
| YOLOv8m-seg | yolov8m_seg | |

전체 목록: `hailomz compile --help`

---

## 성능 참조

### YOLOv5s-seg on Hailo H8 (Profiler)

| 메트릭 | 값 |
|--------|-----|
| Throughput | **92.34 FPS** |
| Latency | **18.40 ms** |
| GOP/s | 2434.09 |
| Input Bandwidth | 108.21 MB/s |
| Output Bandwidth | 591.42 MB/s |

### 클러스터 활용률

| Cluster | Control | Compute | Memory |
|---------|---------|---------|--------|
| cluster_0 | 62.5% | 40.6% | 15.6% |
| cluster_1 | 87.5% | 64.1% | 39.1% |
| cluster_2 | 100% | 87.5% | 53.1% |
| cluster_3 | 87.5% | 64.1% | 44.5% |
| cluster_4 | 43.8% | 31.3% | 60.2% |
| cluster_5 | 93.8% | 43.8% | 23.4% |
| cluster_6 | 68.8% | 45.3% | 32.8% |
| cluster_7 | 43.8% | 34.4% | 35.2% |
| **Total** | **73.4%** | **51.4%** | **38%** |

---

## 산출물 위치

```
HailoRT-Ui/data/models/
├── pt/yolov5s-seg.pt       # PyTorch 모델
├── onnx/yolov5s-seg.onnx   # ONNX 모델
├── har/yolov5s_seg.har     # Hailo Archive
└── hef/yolov5s_seg.hef     # Hailo Executable (최종)
```

---

## 참고 자료

- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
- [Hailo-8 Instance Segmentation](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_instance_segmentation.rst)
- [YOLOv5 Export](https://docs.ultralytics.com/yolov5/tutorials/model_export/)
- [Hailo Community](https://community.hailo.ai/)
