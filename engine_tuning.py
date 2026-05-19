"""
engine_tuning.py — Optuna hyperparameter tuning cho các transformer fraud model.

Cách dùng nhanh:
    python engine_tuning.py --model phobert --dataset samples.jsonl --trials 30
    python engine_tuning.py --model mfinbert --dataset samples.jsonl --trials 30

Cách dùng từ code:
    from engine_tuning import chay_optuna
    best = chay_optuna("samples.jsonl", model_key="mfinbert", n_trials=30)
    print(best.params)

Design:
    - KHÔNG sửa riêng từng engine transformer — hook vào fit_from_split() sẵn có.
    - Mỗi trial: tạo model mới → train trên split → lấy val metric → prune nếu kém.
    - TimeSeriesCV: tao_ts_split() chia fold theo thứ tự thời gian (fold sau = dữ liệu mới hơn).
    - Pruning: MedianPruner cắt trial yếu sớm sau epoch đầu tiên.
    - Kết quả: lưu best_params.json và optuna_study.pkl để reuse.
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from engine_registry import MODEL_PHOBERT, TRANSFORMER_MODEL_CHOICES, lay_lop_mo_hinh

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Search space — chỉnh tại đây, không đụng chỗ khác
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    # Focal Loss
    "gamma":         {"type": "float",  "low": 0.5,   "high": 3.0,   "step": None},
    # Learning rate — log scale vì khoảng cách rất rộng
    "learning_rate": {"type": "float",  "low": 1e-5,  "high": 5e-4,  "log": True},
    # Epochs: int
    "epochs":        {"type": "int",    "low": 2,     "high": 8},
    # Batch size: categorical (phải là bội số của 8 để GPU tận dụng tốt)
    "batch_size":    {"type": "cat",    "choices": [8, 16, 32]},
    # Max sequence length — ảnh hưởng memory nhiều
    "max_len":       {"type": "cat",    "choices": [128, 192, 256]},
}

# Metric tối ưu hoá (phải là key trong dict trả về bởi danh_gia_du_doan)
OPTIMIZE_METRIC = "f2"         # F2 ưu tiên recall — phù hợp fraud detection
OPTIMIZE_DIRECTION = "maximize"

# Time-series CV
DEFAULT_N_FOLDS   = 3          # số fold TS-CV
DEFAULT_N_TRIALS  = 30         # số trial Optuna
DEFAULT_TIMEOUT   = 3600       # giây — dừng sau 1 tiếng dù chưa đủ trials

OUTPUT_BEST_PARAMS_TEMPLATE = "best_params_{model}.json"
OUTPUT_STUDY_PKL_TEMPLATE = "optuna_study_{model}.pkl"


def tao_ten_file_best_params(model_key: str) -> str:
    return OUTPUT_BEST_PARAMS_TEMPLATE.format(model=model_key)


def tao_ten_file_study(model_key: str) -> str:
    return OUTPUT_STUDY_PKL_TEMPLATE.format(model=model_key)


# ---------------------------------------------------------------------------
# Time-Series Split
# ---------------------------------------------------------------------------

@dataclass
class TSSplit:
    """Một fold của time-series CV."""
    fold:        int
    train_texts: list[str]
    train_labels: list[int]
    val_texts:   list[str]
    val_labels:  list[int]

    def summary(self) -> str:
        n_fraud_tr  = sum(self.train_labels)
        n_fraud_val = sum(self.val_labels)
        return (
            f"Fold {self.fold}: train={len(self.train_labels)} "
            f"({n_fraud_tr} fraud) | val={len(self.val_labels)} ({n_fraud_val} fraud)"
        )


def tao_ts_split(
    texts:   list[str],
    labels:  list[int],
    n_folds: int = DEFAULT_N_FOLDS,
    min_train_ratio: float = 0.5,
) -> list[TSSplit]:
    """
    Tạo time-series CV folds.

    Nguyên tắc:
        - Dữ liệu KHÔNG shuffle (giữ thứ tự thời gian).
        - Fold i: train = [0 .. cutoff_i), val = [cutoff_i .. cutoff_i+val_size).
        - Expanding window: mỗi fold train lớn hơn fold trước.

    Ví dụ n=100, n_folds=3:
        Fold 1: train=[0..50), val=[50..67)
        Fold 2: train=[0..67), val=[67..84)
        Fold 3: train=[0..84), val=[84..100)

    Args:
        texts:           danh sách văn bản (đã sắp xếp theo thời gian).
        labels:          nhãn tương ứng.
        n_folds:         số fold.
        min_train_ratio: tỉ lệ tối thiểu train so với toàn bộ data.

    Returns:
        Danh sách TSSplit.
    """
    n = len(texts)
    if n < n_folds * 4:
        raise ValueError(
            f"Dataset quá nhỏ ({n} mẫu) cho {n_folds} folds. "
            f"Cần ít nhất {n_folds * 4} mẫu."
        )

    # Tính các điểm cắt — expanding window
    # Điểm cắt đầu tiên = min_train_ratio * n
    min_train = int(n * min_train_ratio)
    available  = n - min_train                  # phần còn lại để val
    val_size   = available // (n_folds + 1)    # mỗi fold val bằng nhau

    splits: list[TSSplit] = []
    for i in range(n_folds):
        cutoff    = min_train + i * val_size
        val_end   = cutoff + val_size
        if val_end > n:
            val_end = n
        if cutoff >= n or cutoff >= val_end:
            break

        split = TSSplit(
            fold=i + 1,
            train_texts=texts[:cutoff],
            train_labels=labels[:cutoff],
            val_texts=texts[cutoff:val_end],
            val_labels=labels[cutoff:val_end],
        )
        splits.append(split)
        print(f"  {split.summary()}")

    return splits


# ---------------------------------------------------------------------------
# Objective function cho Optuna
# ---------------------------------------------------------------------------

def _sample_params(trial) -> dict:
    """Sample hyperparams từ SEARCH_SPACE cho một trial."""
    import optuna  # import late để không crash khi optuna chưa cài

    params = {}
    for name, cfg in SEARCH_SPACE.items():
        t = cfg["type"]
        if t == "float":
            kwargs = {"log": cfg.get("log", False)}
            if cfg.get("step"):
                kwargs["step"] = cfg["step"]
            params[name] = trial.suggest_float(name, cfg["low"], cfg["high"], **kwargs)
        elif t == "int":
            params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
        elif t == "cat":
            params[name] = trial.suggest_categorical(name, cfg["choices"])
    return params


def tao_objective(
    texts:    list[str],
    labels:   list[int],
    model_key: str = MODEL_PHOBERT,
    n_folds:  int = DEFAULT_N_FOLDS,
    metric:   str = OPTIMIZE_METRIC,
    verbose:  bool = False,
):
    """
    Factory trả về objective function cho optuna.create_study().optimize().

    Mỗi trial:
        1. Sample params.
        2. Chạy time-series CV (n_folds folds).
        3. Trả về mean(metric) qua các folds.
        4. Prune nếu median pruner quyết định.

    Args:
        texts:   toàn bộ texts (theo thứ tự thời gian, KHÔNG shuffle).
        labels:  nhãn tương ứng.
        n_folds: số TS-CV folds.
        metric:  tên metric tối ưu (key trong dict của danh_gia_du_doan).
        verbose: in chi tiết từng fold.
    """
    # Import ở đây để tránh circular import và lazy-load torch
    from engine_common import DataSplit
    import optuna

    model_cls = lay_lop_mo_hinh(model_key)
    ts_splits = tao_ts_split(texts, labels, n_folds=n_folds)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)
        if verbose:
            print(f"\n[Trial {trial.number}] params={params}")

        fold_scores: list[float] = []

        for fold in ts_splits:
            # Tạo model mới với params của trial này
            # Xoá checkpoint cũ nếu có để tránh load nhầm
            ckpt_path = Path(f"tuning_trial_{trial.number}_fold_{fold.fold}.pt")

            try:
                model = model_cls(
                    checkpoint_path  = str(ckpt_path),
                    learning_rate    = params["learning_rate"],
                    epochs           = params["epochs"],
                    batch_size       = params["batch_size"],
                    max_len          = params["max_len"],
                    gamma            = params["gamma"],
                    patience         = 2,   # early stopping nhanh hơn khi tuning
                    threshold_metric = metric,
                )

                # Tạo DataSplit từ fold TS-CV
                split = DataSplit(
                    train_texts  = fold.train_texts,
                    train_labels = fold.train_labels,
                    val_texts    = fold.val_texts,
                    val_labels   = fold.val_labels,
                    # Không cần test set trong tuning — dùng val làm proxy
                    test_texts   = fold.val_texts,
                    test_labels  = fold.val_labels,
                )

                # Train + evaluate trên val của fold này
                val_metrics = model.fit_from_split(split, luu_split=False)
                score = val_metrics.get(metric, 0.0)
                fold_scores.append(score)

                if verbose:
                    print(f"  Fold {fold.fold}: {metric}={score:.4f}")

                # Pruning: báo cáo intermediate value sau mỗi fold
                trial.report(score, step=fold.fold)
                if trial.should_prune():
                    if verbose:
                        print(f"  → Pruned tại fold {fold.fold}")
                    raise optuna.exceptions.TrialPruned()

            finally:
                # Dọn checkpoint tạm sau mỗi fold
                for f in [ckpt_path, ckpt_path.with_suffix(".meta.json"),
                           ckpt_path.with_suffix(".best.pt")]:
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass

        mean_score = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
        if verbose:
            print(f"  → Mean {metric} = {mean_score:.4f}")
        return mean_score

    return objective


# ---------------------------------------------------------------------------
# Main tuning function
# ---------------------------------------------------------------------------

@dataclass
class KetQuaTuning:
    """Kết quả sau khi chạy Optuna."""
    best_params:  dict
    best_value:   float
    metric:       str
    n_trials_done: int
    study_path:   str
    params_path:  str
    all_trials:   list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Best {self.metric}: {self.best_value:.4f}",
            f"Trials completed: {self.n_trials_done}",
            "Best params:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def chay_optuna(
    json_path:   str  = "samples.jsonl",
    model_key:   str  = MODEL_PHOBERT,
    n_trials:    int  = DEFAULT_N_TRIALS,
    n_folds:     int  = DEFAULT_N_FOLDS,
    metric:      str  = OPTIMIZE_METRIC,
    timeout:     Optional[int] = DEFAULT_TIMEOUT,
    study_name:  Optional[str] = None,
    output_dir:  str  = ".",
    verbose:     bool = True,
    load_study:  bool = True,
) -> KetQuaTuning:
    """
    Chạy Optuna hyperparameter tuning với Time-Series CV.

    Args:
        json_path:  đường dẫn file JSONL dataset.
        n_trials:   số trial Optuna muốn chạy.
        n_folds:    số fold TS-CV trong mỗi trial.
        metric:     metric tối ưu (f1, f2, auprc, ...).
        timeout:    dừng sau bao nhiêu giây (None = không giới hạn).
        study_name: tên study Optuna (dùng để resume).
        output_dir: thư mục lưu kết quả.
        verbose:    in chi tiết.
        load_study: nếu True và study pkl tồn tại thì resume thay vì tạo mới.

    Returns:
        KetQuaTuning với best_params và đường dẫn file output.

    Example:
        best = chay_optuna("samples.jsonl", model_key=model_key, n_trials=30)
        print(best)
        # Dùng best params để train lại full:
        model_cls = lay_lop_mo_hinh(model_key)
        model = model_cls(**best.best_params)
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna chưa được cài. Hãy chạy:\n"
            "    pip install optuna"
        )

    if model_key not in TRANSFORMER_MODEL_CHOICES:
        raise ValueError(
            f"Model '{model_key}' không hỗ trợ tuning. "
            f"Chỉ hỗ trợ: {', '.join(TRANSFORMER_MODEL_CHOICES)}"
        )

    model_cls = lay_lop_mo_hinh(model_key)
    if study_name is None:
        study_name = f"{model_key}_fraud_tuning"

    output_path = Path(output_dir)
    study_pkl_path = output_path / tao_ten_file_study(model_key)
    params_json_path = output_path / tao_ten_file_best_params(model_key)

    # Load dataset
    print(f"[Tuning] Đang load dataset: {json_path}")
    texts, labels = model_cls.load_dataset_json(json_path)
    print(
        f"[Tuning] model={model_key} | {len(texts)} mẫu | "
        f"metric={metric} | trials={n_trials} | folds={n_folds}"
    )

    # Tạo hoặc resume study
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=5,   # không prune 5 trial đầu (warm-up)
        n_warmup_steps=1,     # cần ít nhất 1 fold báo cáo trước khi prune
    )
    sampler = optuna.samplers.TPESampler(seed=42)

    if load_study and study_pkl_path.exists():
        print(f"[Tuning] Resume study từ {study_pkl_path}")
        with open(study_pkl_path, "rb") as fh:
            study = pickle.load(fh)
    else:
        study = optuna.create_study(
            study_name  = study_name,
            direction   = OPTIMIZE_DIRECTION,
            pruner      = pruner,
            sampler     = sampler,
        )

    # Tạo objective
    objective = tao_objective(
        texts=texts, labels=labels,
        model_key=model_key, n_folds=n_folds, metric=metric, verbose=verbose,
    )

    print(f"\n[Tuning] Bắt đầu optimize — nhấn Ctrl+C để dừng sớm và lấy best hiện tại\n")
    print("=" * 60)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            gc_after_trial=True,   # giải phóng GPU memory sau mỗi trial
        )
    except KeyboardInterrupt:
        print("\n[Tuning] Dừng sớm theo yêu cầu.")

    # Lưu study để resume sau
    with open(study_pkl_path, "wb") as fh:
        pickle.dump(study, fh)
    print(f"\n[Tuning] Đã lưu study → {study_pkl_path}")

    # Kết quả
    best_trial  = study.best_trial
    best_params = best_trial.params
    best_value  = best_trial.value

    # Lưu best params JSON
    output_payload = {
        "model": model_key,
        "metric":      metric,
        "best_value":  round(best_value, 6),
        "best_params": best_params,
        "n_trials":    len(study.trials),
    }
    params_json_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Tuning] Đã lưu best params → {params_json_path}")

    all_trials = [
        {
            "number": t.number,
            "value":  t.value,
            "state":  str(t.state),
            "params": t.params,
        }
        for t in study.trials
    ]

    result = KetQuaTuning(
        best_params   = best_params,
        best_value    = best_value,
        metric        = metric,
        n_trials_done = len(study.trials),
        study_path    = str(study_pkl_path.resolve()),
        params_path   = str(params_json_path.resolve()),
        all_trials    = all_trials,
    )

    print("\n" + "=" * 60)
    print(result)
    return result


def in_ket_qua_study(study_pkl_path: str) -> None:
    """In lại kết quả của study đã lưu, không cần chạy lại."""
    try:
        import optuna
    except ImportError:
        raise ImportError("pip install optuna")

    with open(study_pkl_path, "rb") as fh:
        study = pickle.load(fh)

    print(f"Study: {study.study_name}")
    print(f"Direction: {study.direction}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Top 5 trials
    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    print("\nTop 5 trials:")
    for t in completed[:5]:
        print(f"  #{t.number:>3}  {study.metric_names[0] if hasattr(study,'metric_names') else 'value'}={t.value:.4f}  {t.params}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _doc_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning cho transformer fraud detection."
    )
    p.add_argument(
        "--model",
        default=MODEL_PHOBERT,
        choices=list(TRANSFORMER_MODEL_CHOICES),
        help="Model transformer cần tuning (phobert hoặc mfinbert).",
    )
    p.add_argument("--dataset",    default="samples.jsonl",  help="File JSONL dataset")
    p.add_argument("--trials",     type=int, default=30,     help="Số trial Optuna")
    p.add_argument("--folds",      type=int, default=3,      help="Số fold Time-Series CV")
    p.add_argument("--metric",     default="f2",             help="Metric tối ưu: f1/f2/auprc/mcc")
    p.add_argument("--timeout",    type=int, default=3600,   help="Timeout giây (0=không giới hạn)")
    p.add_argument("--output-dir", default=".",              help="Thư mục lưu kết quả")
    p.add_argument("--no-resume",  action="store_true",      help="Tạo study mới thay vì resume")
    p.add_argument("--print-study", action="store_true",     help="Chỉ in kết quả study đã lưu")
    p.add_argument("--quiet",      action="store_true",      help="Tắt verbose")
    return p.parse_args()


def main() -> None:
    args = _doc_args()

    if args.print_study:
        pkl = Path(args.output_dir) / tao_ten_file_study(args.model)
        in_ket_qua_study(str(pkl))
        return

    chay_optuna(
        json_path  = args.dataset,
        model_key  = args.model,
        n_trials   = args.trials,
        n_folds    = args.folds,
        metric     = args.metric,
        timeout    = args.timeout if args.timeout > 0 else None,
        output_dir = args.output_dir,
        verbose    = not args.quiet,
        load_study = not args.no_resume,
    )


if __name__ == "__main__":
    main()
