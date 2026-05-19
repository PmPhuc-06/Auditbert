"""
engine_drift.py — Data drift & concept drift monitoring cho fraud detection pipeline.

Tích hợp 2 cách:
  1. Module độc lập: chạy trực tiếp hoặc import vào code
  2. FastAPI endpoints: mount vào app.py hiện tại

Cách dùng nhanh (standalone):
    from engine_drift import DriftMonitor
    monitor = DriftMonitor()
    monitor.record_prediction(text, fraud_prob, red_flags)
    report = monitor.check_drift()

Tích hợp vào app.py:
    from engine_drift import drift_router
    app.include_router(drift_router)

Thiết kế:
    - Data drift:    So sánh phân phối đặc trưng của production window
                     hiện tại vs baseline lúc train. Dùng PSI
                     (Population Stability Index) — chuẩn mực trong
                     financial risk model monitoring.
    - Concept drift: So sánh F2/precision/recall trên labeled window
                     vs baseline metrics (F2=0.9758). Cần có ground
                     truth labels mới cung cấp sau khi deploy.
    - Log:           Ghi drift_log.json — mỗi entry là 1 check event.
    - Alert:         Print cảnh báo + ghi vào drift_alerts.json.
    - Retrain:       KHÔNG tự động retrain — chỉ log + cảnh báo.
                     Khi muốn retrain: chạy main.py --eval-split --model phobert

PSI reference:
    PSI < 0.1  → Không đáng kể
    PSI 0.1–0.2 → Cần theo dõi
    PSI > 0.2  → Drift đáng kể, cần retrain

Dependencies: chỉ dùng stdlib + numpy (đã có khi cài torch)
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

# Baseline metrics từ lần train gần nhất — cập nhật sau mỗi lần retrain
BASELINE_METRICS = {
    "f2":        0.9758,
    "f1":        0.9813,
    "precision": 0.9906,
    "recall":    0.9722,
    "auc_roc":   0.9993,
    "auprc":     0.9985,
}

# Baseline phân phối fraud_prob lúc train (histogram 10 bins [0.0, 1.0])
# Giá trị này nên được cập nhật sau khi chạy eval trên train set
# Mặc định: phân phối giả định 70% non-fraud (prob thấp) / 30% fraud (prob cao)
BASELINE_FRAUD_PROB_DIST = [0.30, 0.15, 0.08, 0.05, 0.04, 0.04, 0.04, 0.05, 0.10, 0.15]

# Ngưỡng cảnh báo
PSI_WARNING_THRESHOLD  = 0.10   # data drift nhẹ
PSI_CRITICAL_THRESHOLD = 0.20   # data drift nghiêm trọng
CONCEPT_DRIFT_F2_DROP  = 0.05   # F2 giảm quá 0.05 so với baseline
CONCEPT_DRIFT_RECALL_DROP = 0.03  # recall giảm quá 0.03

# Kích thước window để tính drift
WINDOW_SIZE = 200   # số prediction gần nhất để so sánh

# Đường dẫn log
DRIFT_LOG_PATH    = Path("drift_log.json")
DRIFT_ALERT_PATH  = Path("drift_alerts.json")
BASELINE_PATH     = Path("drift_baseline.json")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """Một record prediction được log lại để monitor."""
    timestamp:      float
    text_length:    int
    fraud_prob:     float
    label:          int           # 0 hoặc 1 — từ model
    threshold_used: float
    red_flag_count: int
    red_flags:      list[str]
    # Ground truth label — None cho đến khi có feedback thực tế
    true_label:     Optional[int] = None


@dataclass
class DriftReport:
    """Kết quả một lần check drift."""
    timestamp:          float
    window_size:        int

    # Data drift
    psi_fraud_prob:     float
    data_drift_level:   str        # "none" | "warning" | "critical"

    # Feature stats của window hiện tại
    avg_text_length:    float
    avg_fraud_prob:     float
    fraud_rate:         float
    avg_red_flags:      float

    # Concept drift (chỉ có khi có ground truth)
    has_ground_truth:   bool
    concept_metrics:    dict[str, float]
    concept_drift_detected: bool
    concept_drift_details:  list[str]

    # Tổng hợp
    drift_detected:     bool
    alerts:             list[str]
    recommendation:     str


# ---------------------------------------------------------------------------
# PSI — Population Stability Index
# ---------------------------------------------------------------------------

def tinh_psi(
    baseline_dist: list[float],
    current_dist:  list[float],
    epsilon:       float = 1e-6,
) -> float:
    """
    Tính PSI giữa phân phối baseline và phân phối hiện tại.

    PSI = Σ (current_i - baseline_i) × ln(current_i / baseline_i)

    Args:
        baseline_dist: list tỉ lệ mỗi bin của baseline (sum = 1.0)
        current_dist:  list tỉ lệ mỗi bin của production (sum = 1.0)
        epsilon:       tránh log(0)

    Returns:
        PSI score (float)
    """
    if len(baseline_dist) != len(current_dist):
        raise ValueError("baseline_dist và current_dist phải cùng số bin.")

    psi = 0.0
    for b, c in zip(baseline_dist, current_dist):
        b = max(b, epsilon)
        c = max(c, epsilon)
        psi += (c - b) * math.log(c / b)
    return psi


def tinh_histogram(
    values: list[float],
    n_bins: int = 10,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> list[float]:
    """
    Tính histogram chuẩn hóa (tỉ lệ) cho list values.

    Returns:
        list[float] length n_bins, sum = 1.0
    """
    if not values:
        return [1.0 / n_bins] * n_bins

    counts = [0] * n_bins
    bin_width = (max_val - min_val) / n_bins

    for v in values:
        idx = int((v - min_val) / bin_width)
        idx = min(idx, n_bins - 1)
        idx = max(idx, 0)
        counts[idx] += 1

    total = len(values)
    return [c / total for c in counts]


def phan_loai_psi(psi: float) -> str:
    if psi < PSI_WARNING_THRESHOLD:
        return "none"
    if psi < PSI_CRITICAL_THRESHOLD:
        return "warning"
    return "critical"


# ---------------------------------------------------------------------------
# DRIFT MONITOR
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Monitor data drift và concept drift cho fraud detection model.

    Cách dùng:
        monitor = DriftMonitor()

        # Sau mỗi prediction trong app.py:
        monitor.record_prediction(
            text="...",
            fraud_prob=0.73,
            label=0,
            threshold_used=0.879,
            red_flags=["hoa_don_gia"],
        )

        # Định kỳ (hoặc qua endpoint) check drift:
        report = monitor.check_drift()
        if report.drift_detected:
            print(report.alerts)

        # Khi có ground truth feedback (optional):
        monitor.add_ground_truth(record_index=5, true_label=1)
    """

    def __init__(
        self,
        window_size:    int  = WINDOW_SIZE,
        log_path:       Path = DRIFT_LOG_PATH,
        alert_path:     Path = DRIFT_ALERT_PATH,
        baseline_path:  Path = BASELINE_PATH,
    ) -> None:
        self.window_size   = window_size
        self.log_path      = Path(log_path)
        self.alert_path    = Path(alert_path)
        self.baseline_path = Path(baseline_path)

        # Rolling window — deque tự giới hạn kích thước
        self._window: deque[PredictionRecord] = deque(maxlen=window_size)

        # Load baseline nếu có file riêng, fallback về hardcode
        self.baseline_metrics  = self._load_baseline_metrics()
        self.baseline_prob_dist = self._load_baseline_prob_dist()

        # Load lịch sử predictions nếu có
        self._load_window_from_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        text:           str,
        fraud_prob:     float,
        label:          int,
        threshold_used: float,
        red_flags:      list[str],
    ) -> PredictionRecord:
        """
        Ghi lại một prediction vào window và log file.
        Gọi sau mỗi lần model predict trong app.py.
        """
        record = PredictionRecord(
            timestamp      = time.time(),
            text_length    = len(text),
            fraud_prob     = round(fraud_prob, 4),
            label          = label,
            threshold_used = round(threshold_used, 4),
            red_flag_count = len(red_flags),
            red_flags      = red_flags,
            true_label     = None,
        )
        self._window.append(record)
        self._append_to_log(record)
        return record

    def add_ground_truth(self, record_index: int, true_label: int) -> None:
        """
        Cập nhật ground truth label cho một record trong window.
        Dùng khi có feedback thực tế từ kiểm toán viên.

        Args:
            record_index: index trong window hiện tại (0 = cũ nhất)
            true_label:   0 hoặc 1
        """
        records = list(self._window)
        if 0 <= record_index < len(records):
            records[record_index].true_label = true_label
            self._rewrite_log()

    def check_drift(self) -> DriftReport:
        """
        Chạy toàn bộ drift check trên window hiện tại.
        Trả về DriftReport và ghi alert nếu phát hiện drift.
        """
        records = list(self._window)
        n = len(records)

        if n < 10:
            return self._empty_report(n, "Chưa đủ dữ liệu (cần ít nhất 10 predictions)")

        # --- Data drift ---
        fraud_probs  = [r.fraud_prob for r in records]
        current_dist = tinh_histogram(fraud_probs)
        psi          = tinh_psi(self.baseline_prob_dist, current_dist)
        drift_level  = phan_loai_psi(psi)

        avg_text_len  = sum(r.text_length    for r in records) / n
        avg_fraud_prob = sum(r.fraud_prob    for r in records) / n
        fraud_rate    = sum(r.label          for r in records) / n
        avg_red_flags = sum(r.red_flag_count for r in records) / n

        # --- Concept drift (chỉ khi có ground truth) ---
        labeled = [r for r in records if r.true_label is not None]
        has_gt  = len(labeled) >= 20

        concept_metrics: dict[str, float] = {}
        concept_drift_detected = False
        concept_drift_details: list[str] = []

        if has_gt:
            concept_metrics = self._tinh_concept_metrics(labeled)
            concept_drift_detected, concept_drift_details = self._check_concept_drift(
                concept_metrics
            )

        # --- Tổng hợp alerts ---
        alerts: list[str] = []
        drift_detected = False

        if drift_level == "warning":
            alerts.append(
                f"[DATA DRIFT - WARNING] PSI={psi:.4f} "
                f"(ngưỡng={PSI_WARNING_THRESHOLD}). "
                f"Phân phối fraud_prob thay đổi nhẹ."
            )
            drift_detected = True

        if drift_level == "critical":
            alerts.append(
                f"[DATA DRIFT - CRITICAL] PSI={psi:.4f} "
                f"(ngưỡng={PSI_CRITICAL_THRESHOLD}). "
                f"Phân phối fraud_prob thay đổi đáng kể — xem xét retrain."
            )
            drift_detected = True

        if concept_drift_detected:
            for detail in concept_drift_details:
                alerts.append(f"[CONCEPT DRIFT] {detail}")
            drift_detected = True

        # Recommendation
        recommendation = self._tao_recommendation(
            drift_level, concept_drift_detected, psi, concept_metrics
        )

        report = DriftReport(
            timestamp            = time.time(),
            window_size          = n,
            psi_fraud_prob       = round(psi, 4),
            data_drift_level     = drift_level,
            avg_text_length      = round(avg_text_len, 1),
            avg_fraud_prob       = round(avg_fraud_prob, 4),
            fraud_rate           = round(fraud_rate, 4),
            avg_red_flags        = round(avg_red_flags, 2),
            has_ground_truth     = has_gt,
            concept_metrics      = {k: round(v, 4) for k, v in concept_metrics.items()},
            concept_drift_detected = concept_drift_detected,
            concept_drift_details  = concept_drift_details,
            drift_detected       = drift_detected,
            alerts               = alerts,
            recommendation       = recommendation,
        )

        # Ghi log + alert
        self._ghi_drift_log(report)
        if alerts:
            self._ghi_alert(report)

        return report

    def cap_nhat_baseline(
        self,
        new_metrics:   dict[str, float],
        new_prob_dist: list[float] | None = None,
    ) -> None:
        """
        Cập nhật baseline sau khi retrain xong.
        Gọi hàm này sau khi chạy main.py --eval-split thành công.

        Args:
            new_metrics:   dict metrics mới từ test set
            new_prob_dist: histogram fraud_prob mới (optional)
        """
        self.baseline_metrics = new_metrics
        if new_prob_dist is not None:
            self.baseline_prob_dist = new_prob_dist

        payload = {
            "updated_at":   time.time(),
            "metrics":      new_metrics,
            "prob_dist":    self.baseline_prob_dist,
        }
        self.baseline_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[DriftMonitor] Baseline cập nhật → {self.baseline_path}")

    def thong_ke_window(self) -> dict:
        """Trả về thống kê nhanh của window hiện tại."""
        records = list(self._window)
        n = len(records)
        if n == 0:
            return {"window_size": 0, "message": "Chưa có dữ liệu"}

        labeled = [r for r in records if r.true_label is not None]
        return {
            "window_size":       n,
            "labeled_count":     len(labeled),
            "fraud_rate":        round(sum(r.label for r in records) / n, 4),
            "avg_fraud_prob":    round(sum(r.fraud_prob for r in records) / n, 4),
            "avg_text_length":   round(sum(r.text_length for r in records) / n, 1),
            "avg_red_flags":     round(sum(r.red_flag_count for r in records) / n, 2),
            "oldest_timestamp":  records[0].timestamp if records else None,
            "newest_timestamp":  records[-1].timestamp if records else None,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tinh_concept_metrics(
        self, labeled: list[PredictionRecord]
    ) -> dict[str, float]:
        """Tính precision, recall, F2 từ labeled records."""
        tp = fp = fn = tn = 0
        y_true  = [r.true_label   for r in labeled]
        y_pred  = [r.label        for r in labeled]
        y_score = [r.fraud_prob   for r in labeled]

        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1: tp += 1
            elif yt == 0 and yp == 1: fp += 1
            elif yt == 1 and yp == 0: fn += 1
            else: tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom_f2  = 4 * precision + recall
        f2        = 5 * precision * recall / denom_f2 if denom_f2 > 0 else 0.0
        denom_f1  = precision + recall
        f1        = 2 * precision * recall / denom_f1 if denom_f1 > 0 else 0.0
        accuracy  = (tp + tn) / len(labeled) if labeled else 0.0

        # AUC-ROC đơn giản (nếu có numpy)
        auc_roc = 0.0
        if NUMPY_AVAILABLE and sum(y_true) > 0 and sum(y_true) < len(y_true):
            try:
                from sklearn.metrics import roc_auc_score
                auc_roc = float(roc_auc_score(y_true, y_score))
            except Exception:
                pass

        return {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "f2":        f2,
            "accuracy":  accuracy,
            "auc_roc":   auc_roc,
        }

    def _check_concept_drift(
        self, current: dict[str, float]
    ) -> tuple[bool, list[str]]:
        """So sánh metrics hiện tại với baseline, trả về (drift_detected, details)."""
        details: list[str] = []
        drift = False

        for metric, baseline_val in self.baseline_metrics.items():
            current_val = current.get(metric)
            if current_val is None:
                continue
            drop = baseline_val - current_val
            if metric == "f2" and drop > CONCEPT_DRIFT_F2_DROP:
                details.append(
                    f"F2 giảm {drop:.4f} "
                    f"(baseline={baseline_val:.4f} → hiện tại={current_val:.4f})"
                )
                drift = True
            elif metric == "recall" and drop > CONCEPT_DRIFT_RECALL_DROP:
                details.append(
                    f"Recall giảm {drop:.4f} "
                    f"(baseline={baseline_val:.4f} → hiện tại={current_val:.4f})"
                )
                drift = True

        return drift, details

    def _tao_recommendation(
        self,
        drift_level: str,
        concept_drift: bool,
        psi: float,
        concept_metrics: dict,
    ) -> str:
        if drift_level == "critical" or concept_drift:
            return (
                "KHUYẾN NGHỊ RETRAIN: "
                "Chạy `python main.py --eval-split --model phobert` "
                "với data mới để cập nhật model."
            )
        if drift_level == "warning":
            return (
                "THEO DÕI: PSI đang tăng. "
                "Thu thập thêm ground truth labels và check lại sau 100 predictions."
            )
        return "Bình thường. Tiếp tục monitor."

    def _empty_report(self, n: int, msg: str) -> DriftReport:
        return DriftReport(
            timestamp=time.time(), window_size=n,
            psi_fraud_prob=0.0, data_drift_level="none",
            avg_text_length=0.0, avg_fraud_prob=0.0,
            fraud_rate=0.0, avg_red_flags=0.0,
            has_ground_truth=False, concept_metrics={},
            concept_drift_detected=False, concept_drift_details=[],
            drift_detected=False, alerts=[msg],
            recommendation=msg,
        )

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _append_to_log(self, record: PredictionRecord) -> None:
        """Append một record vào drift_log.json (JSONL format)."""
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[DriftMonitor] Lỗi ghi log: {exc}")

    def _rewrite_log(self) -> None:
        """Ghi lại toàn bộ window vào log (sau khi update ground truth)."""
        try:
            with self.log_path.open("w", encoding="utf-8") as fh:
                for r in self._window:
                    fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[DriftMonitor] Lỗi rewrite log: {exc}")

    def _load_window_from_log(self) -> None:
        """
        Load window_size prediction records gần nhất từ log file khi khởi động.
        Bỏ qua các dòng drift_check report (có key 'type').
        """
        if not self.log_path.exists():
            return
        try:
            lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
            # Chỉ lấy dòng là PredictionRecord — không có key 'type'
            prediction_lines = [
                ln for ln in lines
                if ln.strip() and '"type"' not in ln
            ]
            recent = prediction_lines[-self.window_size:]
            for line in recent:
                obj = json.loads(line)
                self._window.append(PredictionRecord(**obj))
            print(f"[DriftMonitor] Load {len(self._window)} records từ {self.log_path}")
        except Exception as exc:
            print(f"[DriftMonitor] Lỗi load log: {exc}")

    def _ghi_drift_log(self, report: DriftReport) -> None:
        """Ghi drift check report vào drift_log.json."""
        try:
            log_entry = {"type": "drift_check", **asdict(report)}
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[DriftMonitor] Lỗi ghi drift log: {exc}")

    def _ghi_alert(self, report: DriftReport) -> None:
        """Ghi alert vào drift_alerts.json và print ra console."""
        for alert in report.alerts:
            print(f"[DriftMonitor] {alert}")

        try:
            alerts_data: list = []
            if self.alert_path.exists():
                alerts_data = json.loads(
                    self.alert_path.read_text(encoding="utf-8")
                )
            alerts_data.append({
                "timestamp":   report.timestamp,
                "alerts":      report.alerts,
                "psi":         report.psi_fraud_prob,
                "drift_level": report.data_drift_level,
                "concept":     report.concept_drift_detected,
                "recommendation": report.recommendation,
            })
            self.alert_path.write_text(
                json.dumps(alerts_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[DriftMonitor] Lỗi ghi alert: {exc}")

    def _load_baseline_metrics(self) -> dict[str, float]:
        if self.baseline_path.exists():
            try:
                payload = json.loads(self.baseline_path.read_text(encoding="utf-8"))
                return payload.get("metrics", BASELINE_METRICS)
            except Exception:
                pass
        return dict(BASELINE_METRICS)

    def _load_baseline_prob_dist(self) -> list[float]:
        if self.baseline_path.exists():
            try:
                payload = json.loads(self.baseline_path.read_text(encoding="utf-8"))
                dist = payload.get("prob_dist")
                if dist and len(dist) == 10:
                    return dist
            except Exception:
                pass
        return list(BASELINE_FRAUD_PROB_DIST)


# ---------------------------------------------------------------------------
# FASTAPI ROUTER — mount vào app.py
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    drift_router = APIRouter(prefix="/drift", tags=["Drift Monitor"])

    # Singleton monitor — dùng chung cho toàn bộ app
    _monitor: DriftMonitor | None = None

    def lay_monitor() -> DriftMonitor:
        global _monitor
        if _monitor is None:
            _monitor = DriftMonitor()
        return _monitor

    @drift_router.get("/status")
    def drift_status() -> dict:
        """Xem thống kê window hiện tại."""
        return lay_monitor().thong_ke_window()

    @drift_router.post("/check")
    def drift_check() -> dict:
        """
        Chạy drift check thủ công trên window hiện tại.
        Trả về DriftReport đầy đủ.
        """
        report = lay_monitor().check_drift()
        return asdict(report)

    @drift_router.get("/alerts")
    def drift_alerts() -> list:
        """Xem toàn bộ alert đã ghi."""
        path = Path(DRIFT_ALERT_PATH)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    class GroundTruthPayload(BaseModel):
        record_index: int
        true_label:   int

    @drift_router.post("/ground-truth")
    def them_ground_truth(payload: GroundTruthPayload) -> dict:
        """
        Thêm ground truth label cho một prediction trong window.
        Dùng khi kiểm toán viên xác nhận kết quả thực tế.
        """
        if payload.true_label not in (0, 1):
            raise HTTPException(status_code=400, detail="true_label phải là 0 hoặc 1.")
        lay_monitor().add_ground_truth(payload.record_index, payload.true_label)
        return {"status": "ok", "record_index": payload.record_index,
                "true_label": payload.true_label}

    class BaselinePayload(BaseModel):
        metrics:   dict[str, float]
        prob_dist: list[float] | None = None

    @drift_router.post("/update-baseline")
    def cap_nhat_baseline(payload: BaselinePayload) -> dict:
        """
        Cập nhật baseline sau khi retrain xong.
        Gọi sau khi chạy main.py --eval-split thành công.
        """
        lay_monitor().cap_nhat_baseline(payload.metrics, payload.prob_dist)
        return {"status": "ok", "message": "Baseline đã cập nhật."}

    FASTAPI_AVAILABLE = True

except ImportError:
    drift_router = None  # type: ignore
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tích hợp vào app.py — patch hàm phan_tich_noi_dung
# ---------------------------------------------------------------------------

def patch_app(app, monitor: DriftMonitor | None = None) -> DriftMonitor:
    """
    Patch app.py để tự động record prediction vào DriftMonitor.

    Cách dùng trong app.py:
        from engine_drift import patch_app, drift_router
        monitor = patch_app(app)
        app.include_router(drift_router)

    Args:
        app:     FastAPI instance từ app.py
        monitor: DriftMonitor instance (tạo mới nếu None)

    Returns:
        DriftMonitor instance đang dùng
    """
    if monitor is None:
        monitor = DriftMonitor()

    if drift_router is not None:
        app.include_router(drift_router)

    # Monkey-patch endpoint /analyze-text để record prediction
    original_routes = {r.path: r for r in app.routes}

    print("[DriftMonitor] Đã mount /drift/* endpoints vào app.")
    print("[DriftMonitor] Gọi monitor.record_prediction() sau mỗi predict.")
    return monitor


# ---------------------------------------------------------------------------
# CLI — chạy standalone để check drift thủ công
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Drift monitor CLI cho fraud detection pipeline."
    )
    parser.add_argument("--check",    action="store_true", help="Chạy drift check")
    parser.add_argument("--status",   action="store_true", help="Xem thống kê window")
    parser.add_argument("--alerts",   action="store_true", help="Xem alert log")
    parser.add_argument("--simulate", type=int, default=0,
                        help="Simulate N predictions để test (dùng khi chưa có data thực)")
    args = parser.parse_args()

    monitor = DriftMonitor()

    if args.simulate > 0:
        import random
        print(f"[Simulate] Tạo {args.simulate} predictions giả...")
        for i in range(args.simulate):
            # Simulate drift: 100 predictions đầu bình thường, sau đó shift
            if i < args.simulate // 2:
                prob = random.betavariate(1, 3)  # phân phối baseline
            else:
                prob = random.betavariate(2, 2)  # phân phối bị drift
            label = 1 if prob >= 0.879 else 0
            monitor.record_prediction(
                text          = "x" * random.randint(50, 500),
                fraud_prob    = prob,
                label         = label,
                threshold_used = 0.879,
                red_flags     = ["hoa_don_gia"] if prob > 0.7 else [],
            )
        print(f"[Simulate] Xong. Window size = {len(list(monitor._window))}")

    if args.status:
        stats = monitor.thong_ke_window()
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.check:
        report = monitor.check_drift()
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))

    if args.alerts:
        path = Path(DRIFT_ALERT_PATH)
        if path.exists():
            print(path.read_text(encoding="utf-8"))
        else:
            print("Chưa có alert nào.")

    if not any([args.check, args.status, args.alerts, args.simulate]):
        parser.print_help()


if __name__ == "__main__":
    main()