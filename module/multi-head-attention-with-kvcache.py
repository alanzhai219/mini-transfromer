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

        self.kv_cache = None
        
    def forward(self, x, mask=None, use_cache=None):
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
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if use_cache and self.kv_cache is not None:
            k_new = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_new = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            # concat k and v
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k_new], dim=2) # (batch_size, num_heads, k_cache_len+seq_len, head_dim)
            v = torch.cat([v_cache, v_new], dim=2) # (batch_size, num_heads, k_cache_len+seq_len, head_dim)
            # update kvcache
            self.kv_cache = (k, v)
        else:
            k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            if use_cache:
                self.kv_cache = (k, v)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.qk_scale  # (batch_size, num_heads, seq_len, seq_len)
        
        # 应用掩码（可选）
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, k.size(2))).unsqueeze(0).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 计算加权值
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 合并多头：(batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 最终线性变换
        output = self.out(attn_output)
        
        if use_cache:
            return output, attn_weights, self.kv_cache
        return output, attn_weights

def generate_example(model, initial_input, max_length):
    """
    使用KV缓存逐token生成序列
    Args:
        model: SelfAttentionWithKVCache或MultiHeadAttentionWithKVCache实例
        initial_input: 初始输入，形状 (batch_size, 1, embed_dim)
        max_length: 最大生成长度
        device: 张量所在设备
    """
    model.eval()
    generated = [initial_input]
    model.kv_cache = None  # 重置缓存
    
    for _ in range(max_length - 1):
        # 获取最新输入
        x = generated[-1]
        # 推理模式，使用KV缓存
        output, attn_weights, kv_cache = model(x, use_cache=True)
        # 模拟生成下一个token（这里仅返回output，实际中可能通过线性层+softmax预测token）
        generated.append(output)
    
    # 拼接所有生成的输出
    generated = torch.cat(generated, dim=1)  # (batch_size, max_length, embed_dim)
    return generated

# 示例使用
if __name__ == "__main__":
    embed_dim = 256
    num_heads = 8
    seq_len = 10
    batch_size = 32

    max_length = 10
    
    # 随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # 多头注意力
    multi_head_attn = MultiHeadAttention(embed_dim, num_heads)
    generated = generate_example(multi_head_attn, x, max_length)
    print(f"Multi-Head Attention (Inference) Generated shape: {generated.shape}")  # (batch_size, max_length, embed_dim)