"""Numerical guards shared by the resident frontend."""

def category_tokens(token_ids, offsets, prompt, spans):
    """Map fixed prompt spans to tokenizer positions without looking at GT."""
    if len(token_ids) != len(offsets):
        raise ValueError("文本 token 与 offset 数量不同")
    groups = [[i for i, (start, end) in enumerate(offsets)
               if end > start and start < hi and end > lo] for lo, hi in spans]
    if any(not indices for indices in groups):
        raise ValueError("类别提示无法映射到 tokenizer")
    return groups


def grounding_probabilities(output, attention_mask):
    """Allow native -inf text padding, but reject invalid real-token scores."""
    import torch
    logits = output.logits
    if (logits.ndim != 3 or attention_mask.ndim != 2 or
            logits.shape[0] != attention_mask.shape[0] or logits.shape[-1] < attention_mask.shape[-1]):
        raise RuntimeError('GroundingDINO 输出与文本 attention_mask 尺寸不一致')
    active = torch.zeros((logits.shape[0], logits.shape[-1]), dtype=torch.bool, device=logits.device)
    active[:, :attention_mask.shape[-1]] = attention_mask.to(device=logits.device, dtype=torch.bool)
    active = active[:, None, :].expand_as(logits)
    if (torch.isnan(logits).any() or torch.isposinf(logits).any() or
            not torch.isfinite(logits[active]).all() or not torch.isfinite(output.pred_boxes).all()):
        raise RuntimeError('GroundingDINO 有效文本分数或预测框输出非有限数值，不能当作无检出计分')
    # Native negative-infinity padding maps to probability zero under sigmoid.
    return logits.sigmoid()
