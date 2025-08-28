import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        """
        自注意力模块（单头）
        Args:
            embed_dim: 输入和输出的嵌入维度
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # 线性层生成查询、键、值
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

        self.qk_scales = torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        
    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: 输入张量，形状 (batch_size, seq_len, embed_dim)
        Returns:
            output: 输出张量，形状 (batch_size, seq_len, embed_dim)
            attn_weights: 注意力权重，形状 (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # 生成q, k, v
        q = self.query(x)  # (batch_size, seq_len, embed_dim)
        k = self.key(x)    # (batch_size, seq_len, embed_dim)
        v = self.value(x)  # (batch_size, seq_len, embed_dim)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.qk_scales

        # 应用掩码（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化，得到注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 计算加权值
        output = torch.matmul(attn_weights, v)  # (batch_size, seq_len, embed_dim)
        
        # 最终线性变换
        output = self.out(output)
        
        return output, attn_weights

# 示例使用
if __name__ == "__main__":
    embed_dim = 256
    seq_len = 10
    batch_size = 32
    
    # 随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # 自注意力
    self_attn = SelfAttention(embed_dim)

    output, attn_weights = self_attn(x)
    print(f"Self-Attention Output shape: {output.shape}")  # (batch_size, seq_len, embed_dim)
    print(f"Self-Attention Weights shape: {attn_weights.shape}")  # (batch_size, seq_len, seq_len)