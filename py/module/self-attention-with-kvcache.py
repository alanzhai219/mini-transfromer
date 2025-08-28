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

        self.kv_cache = None
        
    def forward(self, x, mask=None, use_cache=None):
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

        if use_cache and self.kv_cache is not None:
            k_new = self.key(x)
            v_new = self.value(x)
            # update new token's k and v
            k_cached, v_cached = self.kv_cache
            k = torch.cat([k_cached, k_new], dim=1) # (batch_size, k_cache_len+seq_len, embed_dim)
            v = torch.cat([v_cached, v_new], dim=1) # (batch_size, k_cache_len+seq_len, embed_dim)
            # update kv_cache
            self.kv_cache = (k, v)
        else:
            k = self.key(x)    # (batch_size, seq_len, embed_dim)
            v = self.value(x)  # (batch_size, seq_len, embed_dim)
            if use_cache:
                self.kv_cache = (k, v)
        
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
    seq_len = 10
    batch_size = 32

    max_length = 10
    
    # 随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # 自注意力
    self_attn = SelfAttention(embed_dim)

    generated = generate_example(self_attn, x, max_length)
    print(f"SelfAttention (Inference) Generated shape: {generated.shape}")

    # output, attn_weights = self_attn(x)
    # print(f"Self-Attention Output shape: {output.shape}")  # (batch_size, seq_len, embed_dim)
    # print(f"Self-Attention Weights shape: {attn_weights.shape}")  # (batch_size, seq_len, seq_len)