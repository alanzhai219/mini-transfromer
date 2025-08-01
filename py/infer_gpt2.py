import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel as HF_GPT2LMHeadModel

# Multi-Head Self-Attention Module (Dropout Removed)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Separate Q/K/V linear layers
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)  # Attention output projection

    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        batch_size, seq_len, _ = x.size()

        # Compute Query, Key, Value
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Use KV Cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)

        # Full sequence length including past
        full_seq_len = k.size(-2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            # Ensure attention_mask is [batch_size, 1, 1, full_seq_len]
            if attention_mask.dim() == 2:  # [batch_size, full_seq_len]
                attention_mask = attention_mask[:, None, None, :]  # [batch_size, 1, 1, full_seq_len]
            elif attention_mask.dim() == 4:  # [batch_size, 1, 1, full_seq_len]
                pass  # Already in correct shape
            else:
                raise ValueError(f"Unexpected attention_mask shape: {attention_mask.shape}")
            # Expand to [batch_size, num_heads, seq_len, full_seq_len]
            attention_mask = attention_mask.expand(batch_size, self.num_heads, seq_len, full_seq_len)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, full_seq_len, device=x.device), diagonal=full_seq_len-seq_len+1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 1, float('-inf'))

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Output
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out(out)

        if use_cache:
            return out, (k, v)
        return out, None

# Feed-Forward Network Module (Dropout Removed)
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = F.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x

# Transformer Decoder Layer (Dropout Removed)
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, hidden_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        x = self.ln_1(x)
        attn_output, new_kv = self.attn(x, attention_mask, past_key_value, use_cache)
        x = x + attn_output
        ff_output = self.mlp(x)
        x = x + ff_output
        return x, new_kv

# GPT-2 Model (compatible with distilgpt2, Dropout Removed)
class GPT2LMHeadModel(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12, hidden_dim=3072, num_layers=6, max_length=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Embeddings
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(max_length, embed_dim)

        # Transformer layers
        self.h = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(embed_dim)

        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # Share weights with token embedding

    def forward(self, input_ids, position_ids=None, attention_mask=None, past_key_values=None, use_cache=False):
        batch_size, seq_len = input_ids.size()

        # Default position_ids
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Embeddings
        import pdb; pdb.set_trace()
        token_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        x = token_embeds + position_embeds

        # Transformer layers
        new_past_key_values = [] if use_cache else None
        for i, layer in enumerate(self.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = layer(x, attention_mask, past_kv, use_cache)
            if use_cache:
                new_past_key_values.append(new_kv)

        import pdb; pdb.set_trace()
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, new_past_key_values

    def generate(self, input_ids, attention_mask=None, max_new_tokens=20, do_sample=True, top_k=50):
        batch_size = input_ids.size(0)
        past_key_values = None
        generated_ids = input_ids.clone()
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Update position_ids and input_ids for the current step
                curr_position_ids = position_ids[:, -1:] if step > 0 else position_ids
                curr_input_ids = generated_ids[:, -1:] if step > 0 else generated_ids
                curr_seq_len = curr_input_ids.size(1)

                # Extend attention_mask for the current step
                curr_attention_mask = attention_mask[:, -curr_seq_len:] if step > 0 else attention_mask

                logits, past_key_values = self(
                    input_ids=curr_input_ids,
                    position_ids=curr_position_ids,
                    attention_mask=curr_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                next_token_logits = logits[:, -1, :]
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(top_k_probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, next_token)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                # Debugging: Print next token ID
                print(f"Step {step}: Next token ID = {next_token.item()}")

                # Ensure token ID is within vocabulary range
                if next_token.item() >= self.vocab_size:
                    raise ValueError(f"Generated token ID {next_token.item()} exceeds vocab size {self.vocab_size}")

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                position_ids = torch.cat([position_ids, curr_position_ids + 1], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, 1, device=attention_mask.device)
                ], dim=-1)

        return generated_ids

# Load distilgpt2 weights
def load_distilgpt2_weights(model, pretrained_model_name='distilgpt2'):
    hf_model = HF_GPT2LMHeadModel.from_pretrained(pretrained_model_name)
    state_dict = hf_model.state_dict()

    # Custom state_dict
    custom_state_dict = model.state_dict()

    # Manual weight mapping
    new_state_dict = {}
    for key in state_dict:
        if 'c_attn' in key:
            # Split c_attn weights and biases
            layer_idx = key.split('.')[2]
            if 'weight' in key:
                qkv_weight = state_dict[key]  # [768, 2304]
                q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=-1)  # Each [768, 768]
                new_state_dict[f'h.{layer_idx}.attn.query.weight'] = q_weight.t()  # Transpose to [768, 768]
                new_state_dict[f'h.{layer_idx}.attn.key.weight'] = k_weight.t()
                new_state_dict[f'h.{layer_idx}.attn.value.weight'] = v_weight.t()
            elif 'bias' in key:
                qkv_bias = state_dict[key]  # [2304]
                q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=-1)  # Each [768]
                new_state_dict[f'h.{layer_idx}.attn.query.bias'] = q_bias
                new_state_dict[f'h.{layer_idx}.attn.key.bias'] = k_bias
                new_state_dict[f'h.{layer_idx}.attn.value.bias'] = v_bias
        elif 'c_proj' in key and 'attn' in key:
            # Map attention output projection
            new_key = key.replace('transformer.h.', 'h.').replace('attn.c_proj', 'attn.out')
            new_state_dict[new_key] = state_dict[key]
        elif 'c_fc' in key or 'c_proj' in key:
            # Transpose MLP weights
            new_key = key.replace('transformer.', '')
            new_state_dict[new_key] = state_dict[key].t()  # Transpose to match nn.Linear
        else:
            # Direct mapping for other weights
            new_key = key.replace('transformer.', '')
            new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict, strict=True)
    return model

# Test inference
if __name__ == "__main__":
    # Initialize model
    model = GPT2LMHeadModel(
        vocab_size=50257,
        embed_dim=768,
        num_heads=12,
        hidden_dim=3072,
        num_layers=6,
        max_length=1024
    )
    model = load_distilgpt2_weights(model, 'distilgpt2')
    model.eval()

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

    # Input text
    text = "Once upon a time"
    encoding = tokenizer(text, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # # Debugging: Print initial token IDs
    # print("Initial input_ids:", input_ids.tolist())

    # # Generate text
    # generated_ids = model.generate(
    #     input_ids,
    #     attention_mask,
    #     max_new_tokens=20,
    #     do_sample=True,
    #     top_k=50
    # )
    # print("Generated token IDs:", generated_ids.tolist())
    # print("生成文本:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    # # Example forward pass
    # logits, past_key_values = model(input_ids, attention_mask=attention_mask, use_cache=True)
    # print("Logits 形状:", logits.shape)
    position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)  # tensor([[0, 1, 2, 3]])
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=True)
    outputs_logits = outputs[0] #
    outputs_past_key_values = outputs[1] # 缓存 Key/Value
    import pdb; pdb.set_trace()
    next_token_logits = outputs_logits[:, -1, :]  # 最后一 token 的 logits

    # 预测下一个 token
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # tensor([[in]])
    print(tokenizer.decode(next_token[0]))  # 输出: in

    # 增量生成：只输入新 token
    new_input_ids = next_token  # tensor([[in]])
    new_position_ids = torch.tensor([[4]])  # 下一个位置
    new_attention_mask = torch.ones(1, 5)  # 关注所有之前 token + 新 token

    # 使用 KV Cache 继续生成
    outputs = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids, past_key_values=outputs_past_key_values)
    print(tokenizer.decode(torch.argmax(outputs[0][:, -1, :], dim=-1)))  # 继续预测
