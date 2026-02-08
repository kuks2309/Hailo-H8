# YOLOv5 → Hailo 변환 테스트 비교

> **원본 가이드:** Jig-latch Segmentation 프로젝트 (amap 서버)
> **실제 테스트:** WSL2 환경 테스트 (2026-02-06)

---

## 환경 비교

| 항목 | 원본 가이드 | 실제 테스트 | 차이점 |
|------|------------|------------|--------|
| OS | Ubuntu 22.04 | WSL2 Ubuntu 22.04 | 가상환경 |
| Python | 3.10.12 | 3.10.12 | 동일 |
| PyTorch | 2.3.1+cu118 | 2.10.0+cu128 | 버전 차이 |
| HailoRT | 4.23.0 | 4.23.0 | 동일 |
| hailo-model-zoo | - | 2.17.1 | - |
| ONNX | 1.20.0 | 1.16.0 | **중요: 1.16.0 필수** |
| Hailo 디바이스 | 연결됨 | 미연결 | 벤치마크 제한 |

---

## 모델 정보 비교

| 항목 | 원본 가이드 | 실제 테스트 | 비고 |
|------|------------|------------|------|
| 모델 | YOLOv5s-seg (커스텀) | YOLOv5s-seg (표준) | |
| 클래스 수 | 1 (jig_latch) | 80 (COCO) | |
| 학습 여부 | 500 epochs 학습 | pretrained | |
| PT 파일 크기 | 15 MB | 15.6 MB | 유사 |
| 파라미터 수 | 7,398,422 | 7,611,485 | 클래스 차이 |
| GFLOPs | 25.7 | 26.4 | 유사 |

---

## ONNX Export 비교

| 항목 | 원본 가이드 | 실제 테스트 | 비고 |
|------|------------|------------|------|
| Export 방식 | YOLOv5 공식 export.py | PyTorch 2.x torch.onnx | **차이 발생** |
| opset 버전 | 11 | 18 (자동) | PyTorch 2.x 제한 |
| simplify | 적용 | 적용 (onnxslim) | |
| 출력 shape | (1, 25200, 38) | (1, 25200, 117) | 클래스 수 차이 |
| ONNX 크기 | 28.7 MB | 29.5 MB | 유사 |
| 노드 네이밍 | `/model.24/m.X/Conv` | `conv2d_N` | **비호환** |

### 노드 네이밍 비교 (핵심 차이점)

| 노드 | 원본 가이드 (호환) | 실제 테스트 (비호환) |
|------|-------------------|---------------------|
| Detection Head 1 | `/model.24/m.0/Conv` | `node_conv2d_53` |
| Detection Head 2 | `/model.24/m.1/Conv` | `node_conv2d_54` |
| Detection Head 3 | `/model.24/m.2/Conv` | `node_conv2d_56` |
| Proto Mask | `/model.24/proto/cv3/conv/Conv` | `node_conv2d_62` |

---

## HEF 컴파일 비교

| 항목 | 원본 가이드 | 실제 테스트 | 비고 |
|------|------------|------------|------|
| 컴파일 방법 | hailomz + --end-node-names | hailomz (표준 모델) | |
| ONNX 소스 | 커스텀 export | Hailo Zoo 다운로드 | **해결책** |
| 캘리브레이션 이미지 | train/images | COCO128 (102개) | |
| Mapping 시간 | 1분 21초 | 3분 59초 | CPU 성능 차이 |
| Compilation 시간 | 8초 | 16초 | |
| HEF 크기 | 7.3 MB | 7.4 MB | 동일 |
| HAR 크기 | 146 MB | 151 MB | 유사 |

---

## 성능 비교

| 메트릭 | 원본 가이드 (실측) | 실제 테스트 (프로파일러) | 차이 |
|--------|-------------------|------------------------|------|
| FPS (HW only) | **92.23** | **92.34** | +0.1% |
| FPS (Streaming) | **92.23** | - | 미측정 |
| HW Latency | **18.50 ms** | **18.40 ms** | -0.5% |
| 측정 방법 | `hailo benchmark` | `hailomz profile` | |

### 클러스터 활용률 비교

| Cluster | 원본 가이드 | 실제 테스트 |
|---------|------------|------------|
| cluster_0 | 56.3% / 53.1% / 35.9% | 62.5% / 40.6% / 15.6% |
| cluster_1 | 100% / 67.2% / 68% | 87.5% / 64.1% / 39.1% |
| cluster_2 | 25% / 15.6% / 7% | 100% / 87.5% / 53.1% |
| **Total** | 73.4% / 50.4% / 37.8% | 73.4% / 51.4% / 38% |

> 형식: Control / Compute / Memory

---

## 발견된 이슈 및 해결

### 이슈 1: PyTorch 2.x ONNX 노드 네이밍

| 항목 | 설명 |
|------|------|
| **증상** | `Unable to find end node names: ['Conv_253', 'Conv_232', 'Conv_211']` |
| **원인** | PyTorch 2.x는 `conv2d_N` 형식 노드 이름 생성 |
| **Hailo 기대값** | `/model.24/m.X/Conv` 형식 |
| **해결책** | Hailo Model Zoo 표준 모델 사용 (자동 다운로드) |

### 이슈 2: Opset 버전 다운컨버트 실패

| 항목 | 설명 |
|------|------|
| **증상** | `No Adapter To Version $17 for Resize` |
| **원인** | PyTorch 2.x는 opset 18 사용, 11로 다운컨버트 불가 |
| **해결책** | YOLOv5 공식 export.py 사용 시 정상 작동 |

### 이슈 3: ONNX 버전 충돌

| 항목 | 설명 |
|------|------|
| **증상** | `ModuleNotFoundError: No module named 'onnx.mapping'` |
| **원인** | onnx 1.17+에서 `onnx.mapping` 모듈 제거됨 |
| **해결책** | `pip install onnx==1.16.0` |

---

## 권장 워크플로우

### 표준 모델 (COCO 80 클래스)

```bash
# 1. 캘리브레이션 이미지 준비 (64개 이상)
# 2. 컴파일
hailomz compile yolov5s_seg --hw-arch hailo8 --calib-path ./images --classes 80
# 3. 검증
hailomz profile yolov5s_seg --har yolov5s_seg.har --hw-arch hailo8
```

### 커스텀 모델 (학습된 모델)

```bash
# 1. YOLOv5 학습 환경에서 ONNX export (PyTorch 1.x 권장)
python export.py --weights best.pt --include onnx --opset 11 --simplify

# 2. 노드 이름 확인
python -c "import onnx; m=onnx.load('best.onnx'); print([n.name for n in m.graph.node if n.op_type=='Conv'][-5:])"

# 3. 컴파일
hailomz compile yolov5s_seg \
    --ckpt best.onnx \
    --hw-arch hailo8 \
    --calib-path ./train/images \
    --classes <N> \
    --end-node-names /model.24/m.0/Conv /model.24/m.1/Conv /model.24/m.2/Conv /model.24/proto/cv3/conv/Conv
```

---

## 결론

| 항목 | 원본 가이드 | 실제 테스트 | 결과 |
|------|------------|------------|------|
| ONNX Export | YOLOv5 공식 | PyTorch 2.x | **호환성 이슈** |
| 해결 방법 | - | Hailo Zoo 사용 | **성공** |
| HEF 생성 | ✅ | ✅ | **동일** |
| 예상 성능 | 92 FPS / 18.5ms | 92 FPS / 18.4ms | **동일** |

**핵심 교훈:**
1. PyTorch 2.x 환경에서는 Hailo Model Zoo 표준 모델 사용 권장
2. 커스텀 모델은 YOLOv5 학습 환경에서 직접 export
3. ONNX 버전은 반드시 1.16.0 사용
