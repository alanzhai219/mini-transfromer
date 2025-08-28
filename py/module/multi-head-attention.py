import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        多头注意力模块
        Args:
            embed_dim: 输入和输出的嵌入维度
            num_heads: 注意力头的数量
            dropout: 注意力分数的Dropout比率
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 线性层生成查询、键、值（可共享或每头独立）
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)
        
        self.qk_scale = math.sqrt(self.head_dim)  # 每头的缩放因子
        
    def forward(self, x, mask=None, return_attn_weights=True):
        """
        前向传播
        Args:
            x: 输入张量，形状 (batch_size, seq_len, embed_dim)
            mask: 注意力掩码，形状 (batch_size, num_heads, seq_len, seq_len)，值为0或1
            return_attn_weights: 是否返回注意力权重
        Returns:
            output: 输出张量，形状 (batch_size, seq_len, embed_dim)
            attn_weights: 注意力权重（可选），形状 (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # 生成查询、键、值
        q = self.query(x)  # (batch_size, seq_len, embed_dim)
        k = self.key(x)    # (batch_size, seq_len, embed_dim)
        v = self.value(x)  # (batch_size, seq_len, embed_dim)
        
        # 转换为多头格式：(batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.qk_scale  # (batch_size, num_heads, seq_len, seq_len)
        
        # 应用掩码（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 计算加权值
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 合并多头：(batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 最终线性变换
        output = self.out(attn_output)
        
        if return_attn_weights:
            return output, attn_weights
        return output

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

# 示例使用
if __name__ == "__main__":
    embed_dim = 256
    num_heads = 8
    seq_len = 10
    batch_size = 32
    
    # 随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    causal_mask = create_causal_mask(seq_len)  # 因果掩码
    
    # 多头注意力
    multi_head_attn = MultiHeadAttention(embed_dim, num_heads)
    output, attn_weights = multi_head_attn(x, causal_mask)
    print(f"Multi-Head Attention Output shape: {output.shape}")  # (batch_size, seq_len, embed_dim)
    print(f"Multi-Head Attention Weights shape: {attn_weights.shape}")  # (batch_size, num_heads, seq_len, seq_len)
    
    # 示例：添加位置编码
    class PositionalEncoding(nn.Module):
        def __init__(self, embed_dim, max_len=5000):
            super(PositionalEncoding, self).__init__()
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]
    
    # 应用位置编码
    pos_encoder = PositionalEncoding(embed_dim)
    x_with_pos = pos_encoder(x)
    output, attn_weights = multi_head_attn(x_with_pos, causal_mask)
    print(f"Multi-Head Attention with Pos Encoding shape: {output.shape}")