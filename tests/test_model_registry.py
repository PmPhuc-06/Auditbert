import unittest

from engine_registry import (
    MODEL_BASELINE,
    MODEL_AUDITBERT,
    MODEL_CHOICES,
    MODEL_MFINBERT,
    MODEL_PHOBERT,
    TRANSFORMER_MODEL_CHOICES,
    danh_sach_model_ho_tro,
    la_model_transformer,
    lay_lop_mo_hinh,
)


class ModelRegistryTest(unittest.TestCase):
    def test_registry_exposes_all_supported_models(self) -> None:
        self.assertEqual(
            list(MODEL_CHOICES),
            [MODEL_BASELINE, MODEL_PHOBERT, MODEL_MFINBERT, MODEL_AUDITBERT],
        )
        self.assertEqual(
            danh_sach_model_ho_tro(),
            [MODEL_BASELINE, MODEL_PHOBERT, MODEL_MFINBERT, MODEL_AUDITBERT],
        )

    def test_transformer_registry_marks_mfinbert_correctly(self) -> None:
        self.assertEqual(
            list(TRANSFORMER_MODEL_CHOICES),
            [MODEL_PHOBERT, MODEL_MFINBERT, MODEL_AUDITBERT],
        )
        self.assertFalse(la_model_transformer(MODEL_BASELINE))
        self.assertTrue(la_model_transformer(MODEL_PHOBERT))
        self.assertTrue(la_model_transformer(MODEL_MFINBERT))

    def test_registry_returns_model_classes(self) -> None:
        phobert_cls = lay_lop_mo_hinh(MODEL_PHOBERT)
        mfinbert_cls = lay_lop_mo_hinh(MODEL_MFINBERT)

        self.assertEqual(phobert_cls.__name__, "MoHinhGianLanPhoBERT")
        self.assertEqual(mfinbert_cls.__name__, "MoHinhGianLanMFinBERT")


if __name__ == "__main__":
    unittest.main()
