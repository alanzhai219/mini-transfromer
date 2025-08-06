import torch

# 辅助函数：生成因果掩码（用于解码器）
def create_causal_mask(seq_len):
    """
    生成因果掩码，防止未来信息泄露
    Args:
        seq_len: 序列长度
        device: 张量所在设备
    Returns:
        mask: 掩码张量，形状 (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)
    return mask

# scores = scores.masked_fill(mask.to(scores.device), float('-inf'))