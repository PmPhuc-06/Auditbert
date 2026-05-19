from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import hmac
import os
import re
import secrets
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

from engine_registry import (
    MODEL_AUDITBERT,
    MODEL_BASELINE,
    MODEL_MFINBERT,
    MODEL_PHOBERT,
    MODEL_QUERY_DESCRIPTION,
    danh_sach_model_ho_tro,
    lay_muc_registry,
    lay_lop_mo_hinh,
    tao_mo_hinh_theo_loai,
    tao_ket_qua_du_doan,
)
from engine_common import (
    TRAINING_LABELS,
    TRAINING_TEXTS,
    luu_manifest_split,
    tai_split_tu_file,
    tao_split_mac_dinh_tu_json,
)
from engine_drift import DriftMonitor, drift_router, FASTAPI_AVAILABLE
from engine_document_io import (
    LoiTrichXuatTaiLieu,
    doc_tai_lieu_tu_bytes,
    doc_tai_lieu_tu_duong_dan,
)

LoaiModel = Literal[MODEL_BASELINE, MODEL_PHOBERT, MODEL_MFINBERT, MODEL_AUDITBERT]
MAU_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DangNhapPayload(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def kiem_tra_email(cls, value: str) -> str:
        email = value.strip().lower()
        if not email or not MAU_EMAIL.match(email):
            raise ValueError("Email không hợp lệ.")
        return email

    @field_validator("password")
    @classmethod
    def kiem_tra_mat_khau(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("Mật khẩu phải có ít nhất 8 ký tự.")
        if len(value) > 128:
            raise ValueError("Mật khẩu quá dài.")
        return value


class PhanTichTextPayload(BaseModel):
    text: str
    model: LoaiModel = MODEL_BASELINE

    @field_validator("text")
    @classmethod
    def kiem_tra_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Trường 'text' không được để trống.")
        return text


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class ModelChuaSanSangError(RuntimeError):
    """Model inference chưa sẵn sàng cho API."""


def _phan_tach_model_preload(raw_value: str) -> tuple[str, ...]:
    raw = raw_value.strip().lower()
    if not raw or raw in {"0", "false", "no", "none", "off"}:
        return ()
    return tuple(item.strip().lower() for item in raw_value.split(",") if item.strip())


FRAUD_API_ALLOW_TRAINING = os.getenv("FRAUD_API_ALLOW_TRAINING", "").strip().lower() in {
    "1", "true", "yes", "on",
}
FRAUD_API_PRELOAD_MODELS = _phan_tach_model_preload(
    os.getenv("FRAUD_API_PRELOAD_MODELS", "")
)


def _co_checkpoint_san(model) -> bool:
    checkpoint_path = getattr(model, "checkpoint_path", None)
    return checkpoint_path is not None and Path(checkpoint_path).exists()


def _duong_dan_checkpoint_theo_loai(loai: LoaiModel) -> Path | None:
    checkpoint_path = lay_muc_registry(loai).checkpoint_path
    if not checkpoint_path:
        return None
    return Path(checkpoint_path)


def _thu_load_checkpoint(model) -> bool:
    load_fn = getattr(model, "_load_checkpoint", None)
    if callable(load_fn):
        try:
            return bool(load_fn())
        except Exception as exc:
            print(f"[Startup] Không load được checkpoint: {exc}")
            return False
    return False


def _huan_luyen_model_tu_du_lieu(model, loai: LoaiModel) -> None:
    split = _tai_hoac_tao_split(loai)
    if split is not None:
        model.fit_from_split(
            split,
            luu_split=True,
            ten_file_prefix=_tao_ten_split_prefix(loai),
        )
        return
    if Path(DUONG_DAN_DATASET_MAC_DINH).exists():
        model.fit_from_json(DUONG_DAN_DATASET_MAC_DINH)
        return
    model.fit(TRAINING_TEXTS, TRAINING_LABELS)


def _tao_hoac_tai_model(
    loai: LoaiModel,
    *,
    allow_training: bool,
) -> object:
    checkpoint_path = _duong_dan_checkpoint_theo_loai(loai)
    if checkpoint_path is not None and not checkpoint_path.exists() and not allow_training:
        raise ModelChuaSanSangError(
            f"Model '{loai}' chưa có checkpoint '{checkpoint_path.name}'. "
            f"Hãy train offline bằng CLI trước rồi khởi động lại server, "
            f"hoặc bật FRAUD_API_ALLOW_TRAINING=1 nếu chấp nhận train trong API."
        )

    model = tao_mo_hinh_theo_loai(loai)
    if _co_checkpoint_san(model) and _thu_load_checkpoint(model):
        return model
    if not allow_training:
        raise ModelChuaSanSangError(
            f"Model '{loai}' chưa có checkpoint sẵn cho API. "
            f"Hãy train offline bằng CLI trước rồi khởi động lại server, "
            f"hoặc bật FRAUD_API_ALLOW_TRAINING=1 nếu chấp nhận train trong API."
        )
    _huan_luyen_model_tu_du_lieu(model, loai)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not FRAUD_API_PRELOAD_MODELS:
        print("[Startup] Không preload model nào; API sẽ lazy-load khi có request.")
    for raw_model in FRAUD_API_PRELOAD_MODELS:
        if raw_model not in {MODEL_BASELINE, MODEL_PHOBERT, MODEL_MFINBERT}:
            print(f"[Startup] Bỏ qua model preload không hợp lệ: {raw_model}")
            continue
        loai: LoaiModel = raw_model  # type: ignore[assignment]
        try:
            print(f"[Startup] Đang preload model '{loai}' ...")
            _model_cache[loai] = _tao_hoac_tai_model(
                loai,
                allow_training=FRAUD_API_ALLOW_TRAINING,
            )
            print(f"[Startup] Model '{loai}' sẵn sàng.")
        except Exception as exc:
            print(f"[Startup] Không preload được '{loai}': {exc}")
    yield
    _model_cache.clear()
    print("[Startup] Đã giải phóng model cache.")


app = FastAPI(
    title="Fraud Detection Demo",
    description="API phát hiện gian lận cho văn bản kiểm toán tiếng Việt.",
    version="3.0.0",
    lifespan=lifespan,
)

# Mount drift router
if FASTAPI_AVAILABLE and drift_router is not None:
    app.include_router(drift_router)


# ---------------------------------------------------------------------------
# Khởi tạo model — lazy init
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}
DUONG_DAN_DATASET_MAC_DINH = os.getenv("FRAUD_DATASET_PATH", "samples.jsonl").strip() or "samples.jsonl"

THOI_GIAN_HET_HAN_TOKEN = 3600
EMAIL_DEMO = os.getenv("DEMO_LOGIN_EMAIL", "admin@baoongthay.local").strip().lower()
MAT_KHAU_DEMO = os.getenv("DEMO_LOGIN_PASSWORD", "Admin@123")
TAI_KHOAN_DANG_NHAP = {
    EMAIL_DEMO: hashlib.sha256(MAT_KHAU_DEMO.encode("utf-8")).hexdigest()
}

# Drift monitor — singleton
_drift_monitor: DriftMonitor | None = None


def lay_drift_monitor() -> DriftMonitor:
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor


def _tao_ten_split_prefix(loai: LoaiModel) -> str:
    dataset_path = Path(DUONG_DAN_DATASET_MAC_DINH)
    dataset_key = str(dataset_path.resolve() if dataset_path.exists() else dataset_path)
    dataset_slug = dataset_path.stem or "dataset"
    dataset_hash = hashlib.md5(dataset_key.encode("utf-8")).hexdigest()[:8]
    return f"{loai}_split_{dataset_slug}_{dataset_hash}"


def _tai_hoac_tao_split(loai: LoaiModel):
    split_prefix = _tao_ten_split_prefix(loai)
    split_train = Path(f"{split_prefix}_train.jsonl")
    if split_train.exists():
        return tai_split_tu_file(thu_muc=".", ten_file_prefix=split_prefix)

    if Path(DUONG_DAN_DATASET_MAC_DINH).exists():
        split, manifest = tao_split_mac_dinh_tu_json(
            DUONG_DAN_DATASET_MAC_DINH,
            ty_le_train=0.8,
            ty_le_val=0.1,
            seed=42,
        )
        luu_manifest_split(manifest, f"{split_prefix}_manifest.json")
        return split
    return None


def lay_model(loai: LoaiModel):
    model = _model_cache.get(loai)
    if model is None:
        model = _tao_hoac_tai_model(
            loai,
            allow_training=FRAUD_API_ALLOW_TRAINING,
        )
        _model_cache[loai] = model
    return model


def xac_thuc_dang_nhap(email: str, password: str) -> bool:
    mat_khau_da_bam = TAI_KHOAN_DANG_NHAP.get(email)
    if not mat_khau_da_bam:
        return False
    dau_vao_bam = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(mat_khau_da_bam, dau_vao_bam)


def tao_access_token(email: str) -> str:
    ts = int(time.time())
    nonce = secrets.token_urlsafe(24)
    raw = f"{email}:{ts}:{nonce}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Helper phân tích — tích hợp drift recording
# ---------------------------------------------------------------------------

def phan_tich_noi_dung(text: str, loai: LoaiModel) -> dict:
    try:
        result = _phan_tich_noi_dung_khong_record(text, loai)
    except ModelChuaSanSangError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    prediction = result.pop("_prediction")
    # --- Drift monitoring: record mỗi prediction ---
    try:
        lay_drift_monitor().record_prediction(
            text           = text,
            fraud_prob     = prediction.probability_fraud,
            label          = prediction.label,
            threshold_used = prediction.threshold_used,
            red_flags      = prediction.red_flags,
        )
    except Exception as exc:
        # Không để drift monitor làm crash API
        print(f"[DriftMonitor] Lỗi record prediction: {exc}")

    return result


def _phan_tich_noi_dung_khong_record(text: str, loai: LoaiModel) -> dict:
    try:
        model = lay_model(loai)
    except ModelChuaSanSangError as exc:
        raise exc
    prediction = model.predict(text)
    result = tao_ket_qua_du_doan(
        loai,
        prediction,
        model_source=getattr(model, "model_source", "unknown"),
        top_terms_method=getattr(model, "top_terms_method", "unknown"),
    )
    result["_prediction"] = prediction
    return result


def so_sanh_noi_dung(text: str) -> dict:
    results: list[dict] = []
    ready_models: list[str] = []
    missing_models: list[str] = []

    for loai in danh_sach_model_ho_tro():
        try:
            result = _phan_tich_noi_dung_khong_record(text, loai)  # no drift duplication
            result.pop("_prediction", None)
            result["status"] = "ok"
            results.append(result)
            ready_models.append(loai)
        except ModelChuaSanSangError as exc:
            results.append({
                "model": loai,
                "status": "missing_checkpoint",
                "detail": str(exc),
            })
            missing_models.append(loai)
        except Exception as exc:
            results.append({
                "model": loai,
                "status": "error",
                "detail": str(exc),
            })

    return {
        "comparison_mode": "all_models",
        "ready_models": ready_models,
        "missing_models": missing_models,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def trang_chu() -> str:
    return """
    <html>
      <head>
        <title>Fraud Detection Demo</title>
        <meta charset="utf-8" />
        <style>
          :root {
            color-scheme: light;
            --bg: #f3efe7;
            --paper: #fffaf2;
            --ink: #1d2c2a;
            --accent: #0f766e;
            --accent-soft: #d7f3ee;
            --border: #d6cfbf;
            --danger: #a61b1b;
            --muted: #5a6a67;
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            font-family: "Segoe UI", sans-serif;
            line-height: 1.5;
            color: var(--ink);
            background:
              radial-gradient(circle at top left, #fff8df 0, transparent 28%),
              radial-gradient(circle at bottom right, #d7f3ee 0, transparent 30%),
              var(--bg);
          }
          .page {
            max-width: 1080px;
            margin: 0 auto;
            padding: 32px 20px 48px;
          }
          .hero {
            display: grid;
            gap: 12px;
            margin-bottom: 22px;
          }
          h1 {
            margin: 0;
            font-size: clamp(32px, 4vw, 52px);
            line-height: 1;
            letter-spacing: -0.03em;
          }
          .subtitle {
            max-width: 820px;
            color: var(--muted);
            font-size: 16px;
          }
          .links {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
          }
          .links a, button {
            border: 1px solid var(--border);
            background: var(--paper);
            color: var(--ink);
            text-decoration: none;
            border-radius: 999px;
            padding: 10px 16px;
            font-size: 14px;
            cursor: pointer;
          }
          .links a:hover, button:hover {
            border-color: var(--accent);
          }
          .grid {
            display: grid;
            grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
            gap: 18px;
            align-items: start;
          }
          .panel {
            background: rgba(255, 250, 242, 0.94);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 20px;
            box-shadow: 0 18px 40px rgba(17, 24, 39, 0.08);
            backdrop-filter: blur(6px);
          }
          .panel h2 {
            margin: 0 0 10px;
            font-size: 18px;
          }
          label {
            display: block;
            margin: 14px 0 8px;
            font-weight: 600;
            font-size: 14px;
          }
          select, input[type="file"], textarea {
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 14px;
            background: white;
            font: inherit;
            color: inherit;
          }
          textarea {
            min-height: 140px;
            resize: vertical;
          }
          button {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
            font-weight: 600;
            margin-top: 16px;
          }
          button.secondary {
            background: var(--paper);
            color: var(--ink);
          }
          .hint {
            margin-top: 10px;
            color: var(--muted);
            font-size: 13px;
          }
          .status-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
          }
          .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 13px;
            border: 1px solid var(--border);
            background: white;
          }
          .badge.ready {
            background: var(--accent-soft);
            border-color: #87d4c7;
          }
          .badge.missing {
            background: #fff0f0;
            color: var(--danger);
            border-color: #f1b2b2;
          }
          .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 14px;
          }
          .summary-card {
            border: 1px solid var(--border);
            background: white;
            border-radius: 16px;
            padding: 14px;
          }
          .summary-card strong {
            display: block;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 4px;
          }
          pre {
            margin: 0;
            overflow: auto;
            background: #172120;
            color: #ebf6f4;
            border-radius: 18px;
            padding: 18px;
            font-size: 13px;
            min-height: 280px;
          }
          .error {
            color: var(--danger);
            font-weight: 600;
            margin-top: 12px;
          }
          .mode-toggle {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 8px;
          }
          .mode-toggle button.active {
            background: var(--accent);
            color: white;
          }
          @media (max-width: 900px) {
            .grid {
              grid-template-columns: 1fr;
            }
          }
        </style>
      </head>
      <body>
        <div class="page">
          <section class="hero">
            <h1>Fraud Detection Demo</h1>
            <div class="subtitle">
              Chọn model, bỏ file <code>.txt</code> hoặc <code>.pdf</code> scan/OCR,
              rồi bấm phân tích để demo nhanh cho giảng viên. API đang chạy theo
              chế độ inference-only, nên chỉ những model đã có checkpoint mới sẵn sàng.
            </div>
            <div class="links">
              <a href="/docs" target="_blank" rel="noreferrer">Mo Swagger Docs</a>
              <a href="/redoc" target="_blank" rel="noreferrer">Mo ReDoc</a>
              <a href="/health" target="_blank" rel="noreferrer">Xem Health JSON</a>
            </div>
          </section>

          <div class="grid">
            <section class="panel">
              <h2>Phan Tich Tai Lieu</h2>
              <div id="status">Dang tai trang thai model...</div>
              <div class="mode-toggle">
                <button id="mode-file" type="button" class="active">Tai File</button>
                <button id="mode-text" type="button" class="secondary">Nhap Text</button>
              </div>

              <form id="analyze-form">
                <label for="model">Model</label>
                <select id="model" name="model">
                  <option value="auditbert">&#11088; AuditBERT-VN (Final Model)</option>
                  <option value="baseline">Baseline (TF-IDF)</option>
                  <option value="phobert">PhoBERT</option>
                  <option value="mfinbert">MFinBERT</option>
                </select>

                <div id="file-panel">
                  <label for="file">File .txt hoac .pdf</label>
                  <input id="file" name="file" type="file" accept=".txt,.pdf" />
                  <div class="hint">
                    Ho tro PDF scan/OCR. Nen dung cac bao cao tu cong Kiem toan/Ke toan nha nuoc VN de demo dung boi canh.
                  </div>
                </div>

                <div id="text-panel" style="display:none;">
                  <label for="text">Text can phan tich</label>
                  <textarea id="text" placeholder="Dan noi dung bao cao, doan ngho, hoa don, ket luan kiem toan..."></textarea>
                </div>

                <div class="links" style="margin-top:16px;">
                  <button id="submit-btn" type="submit">Phan Tich Ngay</button>
                  <button id="compare-btn" type="button" class="secondary">So Sanh 3 Model</button>
                </div>
                <div id="error" class="error"></div>
              </form>
            </section>

            <section class="panel">
              <h2>Ket Qua</h2>
              <div id="summary" class="summary"></div>
              <pre id="result">Ket qua JSON se hien thi tai day.</pre>
            </section>
          </div>
        </div>

        <script>
          const statusEl = document.getElementById("status");
          const summaryEl = document.getElementById("summary");
          const resultEl = document.getElementById("result");
          const errorEl = document.getElementById("error");
          const modelEl = document.getElementById("model");
          const fileEl = document.getElementById("file");
          const textEl = document.getElementById("text");
          const filePanel = document.getElementById("file-panel");
          const textPanel = document.getElementById("text-panel");
          const modeFileBtn = document.getElementById("mode-file");
          const modeTextBtn = document.getElementById("mode-text");
          const formEl = document.getElementById("analyze-form");
          const compareBtn = document.getElementById("compare-btn");

          let currentMode = "file";

          function setMode(mode) {
            currentMode = mode;
            const isFile = mode === "file";
            filePanel.style.display = isFile ? "block" : "none";
            textPanel.style.display = isFile ? "none" : "block";
            modeFileBtn.className = isFile ? "active" : "secondary";
            modeTextBtn.className = isFile ? "secondary" : "active";
          }

          function renderSummary(data) {
            const cards = [
              ["Model", data.model || "-"],
              ["Nhan", data.label || "-"],
              ["Rui Ro", data.muc_do_rui_ro || "-"],
              ["Fraud Prob", data.fraud_probability ?? "-"],
              ["Threshold", data.threshold_used ?? "-"],
              ["Nguon", data.source_type || data.file_name || "text"],
            ];
            summaryEl.innerHTML = cards.map(([label, value]) => `
              <div class="summary-card">
                <strong>${label}</strong>
                <span>${value}</span>
              </div>
            `).join("");
          }

          function renderComparison(data) {
            summaryEl.innerHTML = (data.results || []).map((item) => {
              const statusText = item.status === "ok"
                ? `${item.label || "-"} | fraud=${item.fraud_probability ?? "-"}`
                : (item.detail || item.status || "unknown");
              return `
                <div class="summary-card">
                  <strong>${item.model || "-"}</strong>
                  <span>${statusText}</span>
                </div>
              `;
            }).join("");
          }

          function applyCheckpointStatus(health) {
            const checkpointStatus = health.checkpoint_status || {};
            const models = ["baseline", "phobert", "mfinbert"];
            statusEl.innerHTML = `
              <div>Server: <strong>${health.status || "unknown"}</strong></div>
              <div class="status-list">
                ${models.map((model) => {
                  const ready = Boolean(checkpointStatus[model]);
                  const cls = ready ? "badge ready" : "badge missing";
                  const label = ready ? "san sang" : "thieu checkpoint";
                  return `<span class="${cls}">${model}: ${label}</span>`;
                }).join("")}
              </div>
            `;

            let firstReady = null;
            for (const option of modelEl.options) {
              const ready = Boolean(checkpointStatus[option.value]);
              option.disabled = !ready;
              option.textContent = ready
                ? option.value
                : `${option.value} (thieu checkpoint)`;
              if (ready && !firstReady) {
                firstReady = option.value;
              }
            }
            if (firstReady && modelEl.options[modelEl.selectedIndex]?.disabled) {
              modelEl.value = firstReady;
            }
          }

          async function loadHealth() {
            try {
              const response = await fetch("/health");
              const data = await response.json();
              applyCheckpointStatus(data);
            } catch (error) {
              statusEl.innerHTML = `<div class="error">Khong tai duoc /health: ${error.message}</div>`;
            }
          }

          async function submitForm(event) {
            event.preventDefault();
            errorEl.textContent = "";
            resultEl.textContent = "Dang phan tich...";
            summaryEl.innerHTML = "";

            const model = modelEl.value;
            try {
              let response;
              if (currentMode === "file") {
                const file = fileEl.files[0];
                if (!file) {
                  throw new Error("Ban chua chon file.");
                }
                const formData = new FormData();
                formData.append("file", file);
                response = await fetch(`/analyze-file?model=${encodeURIComponent(model)}`, {
                  method: "POST",
                  body: formData,
                });
              } else {
                const text = textEl.value.trim();
                if (!text) {
                  throw new Error("Ban chua nhap text.");
                }
                response = await fetch("/analyze-text", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ text, model }),
                });
              }

              const data = await response.json();
              if (!response.ok) {
                throw new Error(data.detail || JSON.stringify(data));
              }
              renderSummary(data);
              resultEl.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
              errorEl.textContent = error.message || String(error);
              resultEl.textContent = "Khong co ket qua.";
            }
          }

          async function submitComparison() {
            errorEl.textContent = "";
            resultEl.textContent = "Dang so sanh 3 model...";
            summaryEl.innerHTML = "";

            try {
              let response;
              if (currentMode === "file") {
                const file = fileEl.files[0];
                if (!file) {
                  throw new Error("Ban chua chon file.");
                }
                const formData = new FormData();
                formData.append("file", file);
                response = await fetch("/compare-file", {
                  method: "POST",
                  body: formData,
                });
              } else {
                const text = textEl.value.trim();
                if (!text) {
                  throw new Error("Ban chua nhap text.");
                }
                response = await fetch("/compare-text", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ text, model: "baseline" }),
                });
              }

              const data = await response.json();
              if (!response.ok) {
                throw new Error(data.detail || JSON.stringify(data));
              }
              renderComparison(data);
              resultEl.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
              errorEl.textContent = error.message || String(error);
              resultEl.textContent = "Khong co ket qua so sanh.";
            }
          }

          modeFileBtn.addEventListener("click", () => setMode("file"));
          modeTextBtn.addEventListener("click", () => setMode("text"));
          formEl.addEventListener("submit", submitForm);
          compareBtn.addEventListener("click", submitComparison);

          setMode("file");
          loadHealth();
        </script>
      </body>
    </html>
    """


@app.get("/health")
def suc_khoe() -> dict:
    monitor_stats = {}
    try:
        monitor_stats = lay_drift_monitor().thong_ke_window()
    except Exception:
        pass
    checkpoint_status = {
        model_key: (
            checkpoint_path.exists()
            if checkpoint_path is not None
            else False
        )
        for model_key in danh_sach_model_ho_tro()
        for checkpoint_path in [_duong_dan_checkpoint_theo_loai(model_key)]  # noqa: B007
    }
    return {
        "status": "ok",
        "models_available": danh_sach_model_ho_tro(),
        "preload_requested": list(FRAUD_API_PRELOAD_MODELS),
        "preloaded_models": sorted(_model_cache.keys()),
        "checkpoint_status": checkpoint_status,
        "allow_training_in_api": FRAUD_API_ALLOW_TRAINING,
        "drift_monitor": monitor_stats,
    }


@app.post("/login")
def dang_nhap(payload: DangNhapPayload) -> dict:
    try:
        if not xac_thuc_dang_nhap(payload.email, payload.password):
            raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng.")
        return {
            "access_token": tao_access_token(payload.email),
            "token_type": "bearer",
            "expires_in": THOI_GIAN_HET_HAN_TOKEN,
            "user": {"email": payload.email},
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Không thể xử lý đăng nhập lúc này.",
        ) from exc


@app.post("/analyze-text")
def phan_tich_text_api(payload: PhanTichTextPayload) -> dict:
    return phan_tich_noi_dung(payload.text, payload.model)


@app.post("/compare-text")
def so_sanh_text_api(payload: PhanTichTextPayload) -> dict:
    return so_sanh_noi_dung(payload.text)


@app.post("/analyze-file")
async def phan_tich_file_api(
    file: UploadFile = File(...),
    model: LoaiModel = Query(default=MODEL_BASELINE, description=MODEL_QUERY_DESCRIPTION),
) -> dict:
    raw = await file.read()
    try:
        doc = doc_tai_lieu_tu_bytes(raw, file.filename or "")
    except LoiTrichXuatTaiLieu as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    result = phan_tich_noi_dung(doc.text, model)
    result["file_name"] = file.filename
    result["source_type"] = doc.source_type
    result["extraction_method"] = doc.extraction_method
    result["ocr_used"] = doc.ocr_used
    return result


@app.post("/compare-file")
async def so_sanh_file_api(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    try:
        doc = doc_tai_lieu_tu_bytes(raw, file.filename or "")
    except LoiTrichXuatTaiLieu as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    result = so_sanh_noi_dung(doc.text)
    result["file_name"] = file.filename
    result["source_type"] = doc.source_type
    result["extraction_method"] = doc.extraction_method
    result["ocr_used"] = doc.ocr_used
    return result


@app.post("/analyze-pdf")
async def phan_tich_pdf_api(
    file: UploadFile = File(...),
    model: LoaiModel = Query(default=MODEL_BASELINE, description=MODEL_QUERY_DESCRIPTION),
) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .pdf")
    raw = await file.read()
    try:
        doc = doc_tai_lieu_tu_bytes(raw, file.filename)
    except LoiTrichXuatTaiLieu as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    result = phan_tich_noi_dung(doc.text, model)
    result["file_name"] = file.filename
    result["source_type"] = doc.source_type
    result["extraction_method"] = doc.extraction_method
    result["ocr_used"] = doc.ocr_used
    return result


@app.post("/analyze-path")
def phan_tich_duong_dan_api(
    payload: dict,
    model: LoaiModel = Query(default=MODEL_BASELINE, description=MODEL_QUERY_DESCRIPTION),
) -> dict:
    file_path = str(payload.get("file_path", "")).strip()
    if not file_path:
        raise HTTPException(status_code=400, detail="Trường 'file_path' không được để trống.")
    try:
        doc = doc_tai_lieu_tu_duong_dan(file_path)
    except LoiTrichXuatTaiLieu as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    result = phan_tich_noi_dung(doc.text, model)
    result["file_path"] = file_path
    result["source_type"] = doc.source_type
    result["extraction_method"] = doc.extraction_method
    result["ocr_used"] = doc.ocr_used
    return result
