# create_checkpoint.py
from engine_auditbert import MoHinhGianLanAuditBERT
model = MoHinhGianLanAuditBERT(epochs=2)
model.fit_from_json("samples.jsonl")
print("Done. Checkpoint saved at", model.checkpoint_path)