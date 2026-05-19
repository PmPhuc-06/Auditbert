"""
engine_explainability.py — Integrated Gradients explainability cho PhoBERT.

Thay thế attention-based _explain_top_terms() bằng attribution method
đúng chuẩn học thuật (Sundararajan et al., 2017).

Cách dùng (trong MoHinhGianLanPhoBERT):
    from engine_explainability import IntegratedGradientsExplainer
    # Khởi tạo 1 lần trong __init__:
    self.ig_explainer = IntegratedGradientsExplainer(self.model, self.tokenizer, self.device)
    # Gọi trong predict():
    top_terms = self.ig_explainer.explain(text, target_class=1, top_k=5)

Tại sao IG tốt hơn attention:
    - Attention weights KHÔNG phải causal explanation (Jain & Wallace, 2019)
    - IG đo contribution thực sự của từng token bằng cách tích phân gradient
      từ baseline (zero embedding) đến input thực: 
      IG_i = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x−x'))/∂x_i dα
    - Thỏa mãn axiom Completeness: sum(IG) = F(x) - F(baseline)
    - Implementation: xấp xỉ tích phân bằng Riemann sum (n_steps bước)

Dependencies:
    pip install captum
    (torch, transformers đã có sẵn)
"""
from __future__ import annotations

import math
import random
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    from captum.attr import LayerIntegratedGradients
    CAPTUM_AVAILABLE = True
except Exception:
    CAPTUM_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False


# ===========================================================================
# INTEGRATED GRADIENTS EXPLAINER
# ===========================================================================

class IntegratedGradientsExplainer:
    """
    Explainability bằng Layer Integrated Gradients trên embedding layer của PhoBERT.

    Dùng LayerIntegratedGradients (Captum) thay vì IntegratedGradients thuần
    vì PhoBERT nhận token IDs → cần gradient qua embedding layer, không qua
    raw input IDs (discrete → không differentiable trực tiếp).

    Params:
        model:      AutoModelForSequenceClassification đã load và .to(device)
        tokenizer:  AutoTokenizer tương ứng
        device:     torch.device
        n_steps:    số bước xấp xỉ Riemann (cao hơn = chính xác hơn, chậm hơn)
                    50 là đủ cho production; 20 nếu cần nhanh
        internal_batch_size: Captum chia n_steps thành batches nhỏ để tránh OOM
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        n_steps:             int = 50,
        internal_batch_size: int = 10,
    ) -> None:
        if not CAPTUM_AVAILABLE:
            raise RuntimeError(
                "Thiếu Captum. Hãy cài: pip install captum"
            )
        if not TORCH_AVAILABLE:
            raise RuntimeError("Thiếu torch.")

        self.model               = model
        self.tokenizer           = tokenizer
        self.device              = device
        self.n_steps             = n_steps
        self.internal_batch_size = internal_batch_size

        # Hook vào embedding layer của PhoBERT (roberta backbone)
        # LayerIG tính gradient qua layer này thay vì raw input
        embedding_layer = self._get_embedding_layer()
        self.lig = LayerIntegratedGradients(
            self._forward_func,
            embedding_layer,
        )

    def _get_embedding_layer(self):
        """
        Lấy word embedding layer từ model.
        PhoBERT dùng RoBERTa backbone → model.roberta.embeddings.word_embeddings
        Nếu tên khác (DistilBERT, BERT thuần) thì fallback theo thứ tự.
        """
        # Thứ tự ưu tiên cho các backbone phổ biến
        candidates = [
            lambda m: m.roberta.embeddings.word_embeddings,   # RoBERTa / PhoBERT
            lambda m: m.bert.embeddings.word_embeddings,       # BERT
            lambda m: m.distilbert.embeddings.word_embeddings, # DistilBERT
            lambda m: m.base_model.embeddings.word_embeddings, # Generic HF
        ]
        for getter in candidates:
            try:
                layer = getter(self.model)
                print(f"[IG] Embedding layer: {layer.__class__.__name__} "
                      f"(vocab={layer.num_embeddings}, dim={layer.embedding_dim})")
                return layer
            except AttributeError:
                continue
        raise RuntimeError(
            "Không tìm được embedding layer. "
            "Kiểm tra model architecture và cập nhật _get_embedding_layer()."
        )

    def _forward_func(
        self,
        input_ids:      "torch.Tensor",
        attention_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Forward pass trả về logit của target class.
        Captum cần hàm này nhận (embeddings, ...) khi dùng LayerIG.
        
        Lưu ý: LayerIG tự inject embedding — forward_func nhận input_ids
        nhưng Captum override embedding layer bên trong.
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Trả về logits shape [B, 2] — Captum sẽ lấy target class
        return out.logits

    def _build_baseline(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        """
        Baseline = chuỗi toàn [PAD] token, giữ nguyên [CLS] và [SEP].

        Lý do dùng PAD thay vì zero:
        - Zero embedding không tương ứng với token thực nào
        - PAD token được model học để ignore → neutral baseline tốt hơn
        - Giữ [CLS] và [SEP] để không phá cấu trúc sequence
        """
        baseline = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        # Giữ CLS token ở đầu và SEP ở cuối
        baseline[:, 0]  = input_ids[:, 0]   # [CLS]
        # Tìm vị trí [SEP] thực (không phải pad)
        sep_id = self.tokenizer.sep_token_id
        for i in range(input_ids.size(0)):
            sep_positions = (input_ids[i] == sep_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                baseline[i, sep_positions[0]] = sep_id
        return baseline

    def explain(
        self,
        text:         str,
        target_class: int = 1,
        top_k:        int = 5,
        max_len:      int = 256,
    ) -> list[dict]:
        """
        Tính Integrated Gradients attribution cho từng token trong text.

        Args:
            text:         văn bản cần giải thích
            target_class: 0 = non-fraud, 1 = fraud (default)
            top_k:        số token quan trọng nhất trả về
            max_len:      max sequence length (phải khớp với lúc train)

        Returns:
            list[dict] mỗi phần tử gồm:
                - token:       subword token string
                - word:        word gốc (ghép từ subwords)
                - attribution: float, dương = ủng hộ target_class,
                               âm = chống lại target_class
                - abs_score:   |attribution| để sort

        Raises:
            RuntimeError nếu Captum không có hoặc model chưa train.
        """
        self.model.eval()

        # Tokenize
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        baseline_ids   = self._build_baseline(input_ids)

        # Tính attribution
        # LayerIG trả về attributions shape [1, seq_len, embed_dim]
        # Summing theo embed_dim → scalar per token
        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target_class,
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
            return_convergence_delta=True,
        )

        # Sum embedding dim → [seq_len] attribution per token
        attr_sum = attributions.squeeze(0).sum(dim=-1)  # [seq_len]

        # Completeness check — delta nhỏ = xấp xỉ tốt
        delta_val = delta.abs().mean().item()
        if delta_val > 0.05:
            print(f"[IG] Cảnh báo: convergence delta={delta_val:.4f} "
                  f"(>0.05). Tăng n_steps để chính xác hơn.")

        # Lấy tokens
        tokens    = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        attr_list = attr_sum.detach().cpu().tolist()

        # Lọc special tokens
        special = {
            self.tokenizer.cls_token,
            self.tokenizer.sep_token,
            self.tokenizer.pad_token,
            "<s>", "</s>", "<pad>", "<unk>",
        }

        # Ghép subwords thành words (PhoBERT dùng "▁" prefix cho word-start)
        # và tổng hợp attribution theo word
        word_attributions: list[dict] = []
        current_word       = ""
        current_attr_sum   = 0.0
        current_tokens:  list[str] = []

        for token, attr in zip(tokens, attr_list):
            if token in special:
                continue
            # PhoBERT / RoBERTa: "▁" đánh dấu đầu word mới
            is_word_start = token.startswith("▁") or not current_word
            if is_word_start and current_word:
                word_attributions.append({
                    "word":        current_word,
                    "tokens":      current_tokens[:],
                    "attribution": current_attr_sum,
                    "abs_score":   abs(current_attr_sum),
                })
                current_word     = token.lstrip("▁")
                current_attr_sum = attr
                current_tokens   = [token]
            else:
                current_word    += token.lstrip("▁")
                current_attr_sum += attr
                current_tokens.append(token)

        # Flush word cuối
        if current_word:
            word_attributions.append({
                "word":        current_word,
                "tokens":      current_tokens,
                "attribution": current_attr_sum,
                "abs_score":   abs(current_attr_sum),
            })

        # Lọc token quá ngắn hoặc không phải chữ
        filtered = [
            w for w in word_attributions
            if len(w["word"]) >= 2
            and len(w["word"]) <= 30          # loại chuỗi số/token khổng l                        
            and "@@" not in w["word"]         # loại subword chưa ghép đúng
            and any(c.isalpha() for c in w["word"])
            and not all(c.isdigit() or c in ".,: " for c in w["word"])  # loại chuỗi số thuầ
        ]

        # Sort theo abs_score, lấy top_k
        filtered.sort(key=lambda x: x["abs_score"], reverse=True)
        top = filtered[:top_k]

        # Normalize attribution về [-1, 1] để dễ đọc
        max_abs = max((w["abs_score"] for w in top), default=1.0) or 1.0
        for w in top:
            w["attribution_normalized"] = round(w["attribution"] / max_abs, 4)
            w["attribution"]            = round(w["attribution"], 6)
            w["abs_score"]              = round(w["abs_score"], 6)

        return top

    def explain_top_terms(
        self,
        text:     str,
        top_k:    int = 5,
        max_len:  int = 256,
    ) -> list[str]:
        """
        Wrapper trả về list[str] — tương thích với interface hiện tại
        của MoHinhGianLanPhoBERT.predict() (trường top_terms).

        Dấu (+) = ủng hộ fraud, (-) = chống lại fraud.
        """
        try:
            results = self.explain(text, target_class=1, top_k=top_k, max_len=max_len)
            terms = []
            for r in results:
                sign  = "+" if r["attribution"] >= 0 else "-"
                score = abs(r["attribution_normalized"])
                terms.append(f"{r['word']}({sign}{score:.2f})")
            return terms
        except Exception as exc:
            print(f"[IG] Lỗi explain: {exc}. Trả về rỗng.")
            return []

    def explain_html(
        self,
        text:    str,
        top_k:   int = 10,
        max_len: int = 256,
    ) -> str:
        """
        Trả về HTML highlight văn bản theo attribution score.
        Token đỏ = ủng hộ fraud, xanh = chống lại fraud.
        Độ đậm tỉ lệ với |attribution|.

        Dùng cho API endpoint /explain hoặc frontend visualization.
        """
        try:
            results = self.explain(text, target_class=1, top_k=top_k, max_len=max_len)
        except Exception as exc:
            return f"<span>Lỗi explain: {exc}</span>"

        # Map word → attribution normalized
        word_attr = {r["word"].lower(): r["attribution_normalized"] for r in results}

        # Tokenize text đơn giản để giữ spacing/punctuation
        import re
        tokens = re.findall(r"\S+|\s+", text)

        parts = []
        for tok in tokens:
            clean = re.sub(r"[^\w]", "", tok).lower()
            if clean in word_attr:
                score = word_attr[clean]
                if score > 0:
                    # Đỏ — ủng hộ fraud
                    alpha = min(0.85, 0.2 + abs(score) * 0.65)
                    style = f"background:rgba(220,50,50,{alpha:.2f});border-radius:3px;padding:1px 2px"
                else:
                    # Xanh — chống lại fraud
                    alpha = min(0.85, 0.2 + abs(score) * 0.65)
                    style = f"background:rgba(30,150,80,{alpha:.2f});border-radius:3px;padding:1px 2px"
                parts.append(f'<span style="{style}" title="IG={score:+.3f}">{tok}</span>')
            else:
                parts.append(tok)

        body = "".join(parts)
        legend = (
            '<div style="font-size:11px;margin-top:8px;color:#666">'
            '<span style="background:rgba(220,50,50,0.5);padding:1px 6px;border-radius:3px">■</span> Ủng hộ fraud &nbsp;'
            '<span style="background:rgba(30,150,80,0.5);padding:1px 6px;border-radius:3px">■</span> Chống lại fraud'
            '</div>'
        )
        return f'<div style="line-height:1.8;font-size:14px">{body}</div>{legend}'


class PerturbationTextExplainer:
    """
    Text explainer xấp xỉ cho SHAP/LIME bằng perturbation trên token.

    Đây là bản lightweight, không phụ thuộc package shap/lime bên ngoài.
    Mục tiêu là bổ sung insight phục vụ demo/đánh giá, không thay thế
    implementation SHAP/LIME đầy đủ trong nghiên cứu production.
    """

    def __init__(self, predict_score_batch) -> None:
        self.predict_score_batch = predict_score_batch

    def _tokenize_words(self, text: str, max_tokens: int = 12) -> list[str]:
        tokens = re.findall(r"[A-Za-zÀ-ỹà-ỹ0-9_]+", text, flags=re.UNICODE)
        return tokens[:max_tokens]

    def _format_terms(self, scores_by_token: list[tuple[str, float]], top_k: int) -> list[str]:
        ranked = sorted(scores_by_token, key=lambda item: abs(item[1]), reverse=True)[:top_k]
        results: list[str] = []
        for token, score in ranked:
            sign = "+" if score >= 0 else "-"
            results.append(f"{token}({sign}{abs(score):.2f})")
        return results

    def explain_shap_approx(self, text: str, top_k: int = 5) -> list[str]:
        tokens = self._tokenize_words(text)
        if not tokens:
            return []

        full_score = float(self.predict_score_batch([text])[0])
        contributions: list[tuple[str, float]] = []
        for index, token in enumerate(tokens):
            perturbed_tokens = tokens[:index] + tokens[index + 1:]
            perturbed_text = " ".join(perturbed_tokens) if perturbed_tokens else ""
            perturbed_score = float(self.predict_score_batch([perturbed_text])[0])
            contributions.append((token, full_score - perturbed_score))
        return self._format_terms(contributions, top_k=top_k)

    def explain_lime_surrogate(
        self,
        text: str,
        top_k: int = 5,
        num_samples: int = 48,
        seed: int = 42,
    ) -> list[str]:
        tokens = self._tokenize_words(text)
        if not tokens:
            return []
        if not NUMPY_AVAILABLE:
            return self.explain_shap_approx(text, top_k=top_k)

        rng = random.Random(seed)
        sample_count = max(num_samples, len(tokens) + 2)
        masks = [[1] * len(tokens)]
        while len(masks) < sample_count:
            mask = [1 if rng.random() > 0.35 else 0 for _ in tokens]
            if sum(mask) == 0:
                mask[rng.randrange(len(tokens))] = 1
            masks.append(mask)

        perturbed_texts = []
        weights = []
        for mask in masks:
            selected = [token for token, keep in zip(tokens, mask) if keep]
            perturbed_texts.append(" ".join(selected))
            distance = len(tokens) - sum(mask)
            kernel_width = max(len(tokens) * 0.75, 1.0)
            weights.append(math.exp(-((distance ** 2) / (kernel_width ** 2))))

        y = np.array(self.predict_score_batch(perturbed_texts), dtype=float)
        x = np.array(masks, dtype=float)
        ones = np.ones((x.shape[0], 1), dtype=float)
        design = np.concatenate([ones, x], axis=1)
        weight_matrix = np.diag(np.array(weights, dtype=float))
        beta = np.linalg.pinv(design.T @ weight_matrix @ design) @ design.T @ weight_matrix @ y
        token_scores = [(token, float(weight)) for token, weight in zip(tokens, beta[1:].tolist())]
        return self._format_terms(token_scores, top_k=top_k)
