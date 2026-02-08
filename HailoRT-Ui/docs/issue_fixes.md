# Issue Fixes - HailoRT-Ui

**Date:** 2026-02-08
**Review source:** Automated code review (Claude Opus 4.6)

---

## Fix 1: Cooperative Cancellation for ConvertWorker

**Issues:** HIGH-1, HIGH-5
**Severity:** HIGH
**Files modified:**
- `src/workers/convert_worker.py`
- `src/services/converter_service.py`
- `src/app.py`

### Problem

`QThread.terminate()` was used to stop conversion workers on application exit. This forcibly kills the thread at the OS level without executing cleanup code, destructors, or `finally` blocks. This could:
- Leave mutexes locked
- Corrupt output files mid-write (`.onnx`, `.hef`)
- Cause resource leaks

Additionally, `ConvertWorker` had no cancellation mechanism at all -- long-running operations like PyTorch export or Hailo SDK quantization could not be interrupted gracefully.

### Solution

**ConvertWorker (`convert_worker.py`):**
- Added `self._cancelled` flag and `cancel()` method for cooperative cancellation
- Added `is_cancelled` property for external status checks
- Added `_check_cancelled()` helper that raises `InterruptedError`
- Cancellation checks inserted before and after major conversion steps
- `InterruptedError` caught separately to emit clean "Cancelled by user" status

**ConverterService (`converter_service.py`):**
- Added `cancel_cb` parameter to `set_callbacks()`
- Added `_check_cancelled()` method that calls the cancel callback
- Cancellation checks inserted at key points in `convert_pt_to_onnx()` and `compile_onnx_to_hef()`:
  - After model loading
  - Before export/compilation dispatch
  - Before hailomz CLI fallback

**App (`app.py`):**
- `closeEvent()` now calls `worker.cancel()` first with 10s wait
- Falls back to `worker.terminate()` only if cooperative cancellation times out
- Added warning log when force-terminate is needed

### Before
```python
# app.py
worker.terminate()
worker.wait(5000)
```

### After
```python
# app.py
worker.cancel()
worker.wait(10000)
if worker.isRunning():
    logger.warning("ConvertWorker did not stop gracefully, forcing terminate")
    worker.terminate()
    worker.wait(3000)
```

---

## Fix 2: QImage Buffer Lifetime (Dangling Pointer)

**Issue:** MEDIUM-5
**Severity:** MEDIUM
**File modified:** `src/views/inference_tab.py` (line 404)

### Problem

`QImage` constructed from `numpy.ndarray.data` does NOT copy the underlying memory buffer. When the local `resized` variable goes out of scope at the end of `_display_frame()`, the numpy array's buffer may be freed while the `QImage`/`QPixmap` is still referencing it. This could cause:
- Corrupted frame display
- Sporadic segfaults under high frame rates

This is a well-known PyQt5 footgun documented in Qt: "The buffer must remain valid throughout the life of the QImage."

### Solution

Added `.copy()` to force a deep copy of the QImage, decoupling it from the numpy buffer lifetime.

### Before
```python
q_img = QImage(resized.data, w, h, bytes_per_line, QImage.Format_RGB888)
```

### After
```python
q_img = QImage(resized.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
```

---

## Fix 3: MonitorTab Orphaned HailoService via Dependency Injection

**Issue:** HIGH-4
**Severity:** HIGH
**Files modified:**
- `src/views/monitor_tab.py`
- `src/app.py`

### Problem

Every 1-second timer tick in `MonitorTabController._update_stats()` created a new `HailoService()` instance. This new instance:
1. Never called `connect()`, so `self.device` was always `None`
2. `get_device_info()` always returned an empty dict `{}`
3. Real device data was never displayed in the monitor tab
4. The `DeviceTabController` maintained its own connected `HailoService`, but the monitor tab couldn't access it

### Solution

**MonitorTabController (`monitor_tab.py`):**
- Added `device_controller` parameter to `__init__()`
- `_update_stats()` now checks if the shared `device_controller` has an active connection
- If connected: queries the existing `hailo_service` instance for real device stats
- If not connected: falls back to mock stats (same behavior as before for mock mode)
- Removed per-tick `HailoService()` instantiation entirely

**App (`app.py`):**
- Passes `device_controller=self.device_controller` to `MonitorTabController` constructor
- Monitor tab now shares the same device connection as the device tab

### Before
```python
# monitor_tab.py
def _update_stats(self) -> None:
    from services.hailo_service import HailoService
    service = HailoService()  # New instance every tick, never connected
    info = service.get_device_info()  # Always returns {}

# app.py
self.monitor_controller = MonitorTabController(self.tab_monitor, self.base_path)
```

### After
```python
# monitor_tab.py
def _update_stats(self) -> None:
    if (self.device_controller
            and self.device_controller.hailo_service
            and self.device_controller.is_connected):
        info = self.device_controller.hailo_service.get_device_info()  # Real data
        self._update_device_stats(info)
    else:
        self._update_mock_stats()

# app.py
self.monitor_controller = MonitorTabController(
    self.tab_monitor, self.base_path,
    device_controller=self.device_controller
)
```

---

## Summary

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | ConvertWorker cooperative cancellation (HIGH-1, HIGH-5) | HIGH | Fixed |
| 2 | QImage buffer lifetime (MEDIUM-5) | MEDIUM | Fixed |
| 3 | MonitorTab orphaned HailoService (HIGH-4) | HIGH | Fixed |

All modified files pass `py_compile` verification.
