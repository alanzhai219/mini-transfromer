import torch
import math

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention_ref(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    # attn_weight = query @ key.transpose(-2, -1) * scale_factor
    import pdb; pdb.set_trace()
    key_trans = key.transpose(-2, -1) * scale_factor
    attn_weight = torch.matmul(query, key_trans)
    # attn_weight = torch.matmul(query, key.transpose(-2, -1) * scale_factor)
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return torch.matmul(attn_weight, value)


query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 32, dtype=torch.float16, device="cuda")

result0 = scaled_dot_product_attention_ref(query, key, value)
result1 = torch.nn.functional.scaled_dot_product_attention(query, key, value)

print(result0.shape) # [32, 8, 128, 64]
print(result0.flatten()[:10])
print(result1.flatten()[:10])
