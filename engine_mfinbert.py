from __future__ import annotations

from engine_common import (
    MFINBERT_BATCH_SIZE,
    MFINBERT_CHECKPOINT,
    MFINBERT_EPOCHS,
    MFINBERT_LR,
    MFINBERT_MAX_LEN,
    MFINBERT_MODEL_NAME,
    MFINBERT_PATIENCE,
    MFINBERT_THRESHOLD_METRIC,
)
from engine_transformer import FOCAL_GAMMA_DEFAULT, MoHinhGianLanTransformer


class MoHinhGianLanMFinBERT(MoHinhGianLanTransformer):
    """
    Wrapper MFinBERT trên core transformer fraud model dùng chung.
    """

    DISPLAY_NAME = "MFinBERT"

    def __init__(
        self,
        model_name: str = MFINBERT_MODEL_NAME,
        checkpoint_path: str = MFINBERT_CHECKPOINT,
        max_len: int = MFINBERT_MAX_LEN,
        batch_size: int = MFINBERT_BATCH_SIZE,
        epochs: int = MFINBERT_EPOCHS,
        learning_rate: float = MFINBERT_LR,
        threshold_metric: str = MFINBERT_THRESHOLD_METRIC,
        patience: int = MFINBERT_PATIENCE,
        hybrid_metadata_enabled: bool = False,
        gamma: float = FOCAL_GAMMA_DEFAULT,
        ig_n_steps: int = 50,
        ig_internal_batch_size: int = 10,
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
        )
