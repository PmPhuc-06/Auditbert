from __future__ import annotations

from engine_common import (
    PHOBERT_BATCH_SIZE,
    PHOBERT_CHECKPOINT,
    PHOBERT_EPOCHS,
    PHOBERT_LR,
    PHOBERT_MAX_LEN,
    PHOBERT_MODEL_NAME,
    PHOBERT_PATIENCE,
    PHOBERT_THRESHOLD_METRIC,
)
from engine_transformer import FOCAL_GAMMA_DEFAULT, MoHinhGianLanTransformer


class MoHinhGianLanPhoBERT(MoHinhGianLanTransformer):
    """
    Wrapper PhoBERT trên core transformer fraud model dùng chung.
    """

    DISPLAY_NAME = "PhoBERT"

    def __init__(
        self,
        model_name: str = PHOBERT_MODEL_NAME,
        checkpoint_path: str = PHOBERT_CHECKPOINT,
        max_len: int = PHOBERT_MAX_LEN,
        batch_size: int = PHOBERT_BATCH_SIZE,
        epochs: int = PHOBERT_EPOCHS,
        learning_rate: float = PHOBERT_LR,
        threshold_metric: str = PHOBERT_THRESHOLD_METRIC,
        patience: int = PHOBERT_PATIENCE,
        hybrid_metadata_enabled: bool = False,
        gamma: float = FOCAL_GAMMA_DEFAULT,
        ig_n_steps: int = 50,
        ig_internal_batch_size: int = 10,
        freeze_layers: int = 8,   # Fix collapse: đóng băng 8/12 layers
    ) -> None:
        super().__init__(
            display_name=self.DISPLAY_NAME,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            max_len=max_len,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            threshold_metric=threshold_metric,
            patience=patience,
            hybrid_metadata_enabled=hybrid_metadata_enabled,
            gamma=gamma,
            ig_n_steps=ig_n_steps,
            ig_internal_batch_size=ig_internal_batch_size,
            freeze_layers=freeze_layers,
        )
