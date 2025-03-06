import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Step 1: Define the multi-head attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # Reshape from (batch_size, seq_len, d_model) to (batch_size, num_heads, seq_len, d_k)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections and split into heads
        q = self.split_heads(self.w_q(q), batch_size)  # (batch_size, num_heads, seq_len_q, d_k)
        k = self.split_heads(self.w_k(k), batch_size)  # (batch_size, num_heads, seq_len_k, d_k)
        v = self.split_heads(self.w_v(v), batch_size)  # (batch_size, num_heads, seq_len_v, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = scores / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len_q, d_k)

        # Reshape back to (batch_size, seq_len_q, d_model)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.w_o(context)

        return output

# Step 2: Define the position-wise feed-forward network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# Step 3: Define positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()

        # Create a positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register the positional encoding as a buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Add positional encoding to the input
        return x + self.pe[:, :x.size(1)]

# Step 4: Define encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x

# Step 5: Define decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection and layer normalization
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Cross-attention with residual connection and layer normalization
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x

# Step 6: Define the transformer encoder
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Step 7: Define the transformer decoder
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

# Step 8: Define the full transformer model for language modeling (decoder-only architecture)
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerLM, self).__init__()

        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # Decoder
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Create embeddings and apply positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through decoder (using x as both input and target)
        # This is a decoder-only architecture common in language models like GPT
        x = self.decoder(x, x, None, mask)

        # Project to vocabulary
        output = self.fc_out(x)

        return output

# Step 9: Create a causal (triangular) mask for autoregressive generation
def create_causal_mask(seq_len):
    """Create a causal attention mask to prevent attending to future tokens."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # Convert to boolean mask where True values are kept

# Step 10: Define a simple tokenizer for text
class SimpleTokenizer:
    def __init__(self, texts=None):
        self.char_to_idx = {}
        self.idx_to_char = {}

        if texts:
            self.fit(texts)

    def fit(self, texts):
        # Create a set of all unique characters
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)

        # Create mapping dictionaries
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(unique_chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def encode(self, text):
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices])

    @property
    def vocab_size(self):
        return len(self.char_to_idx)

# Step 11: Define a dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # Tokenize all texts and concatenate
        self.data = []
        for text in texts:
            self.data.extend(tokenizer.encode(text))

        # Calculate the number of sequences we can create
        self.num_sequences = max(0, len(self.data) - seq_length)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Get input sequence and target (next tokens)
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )

# Step 12: Training function
def train_transformer_lm(model, dataloader, num_epochs, learning_rate, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Create causal mask
            seq_len = inputs.size(1)
            causal_mask = create_causal_mask(seq_len).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs, causal_mask)

            # Calculate loss
            # Reshape for cross-entropy: (batch_size * seq_len, vocab_size)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, '
                      f'Loss: {total_loss / (batch_idx + 1):.4f}')

        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(dataloader):.4f}')

    return model

# Step 13: Text generation function
def generate_text(model, tokenizer, start_text, max_length, temperature=1.0, device='cpu'):
    model.eval()

    # Encode the start text
    input_seq = tokenizer.encode(start_text)
    input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

    # Generate new tokens one by one
    for _ in range(max_length):
        # Create causal mask
        seq_len = input_tensor.size(1)
        causal_mask = create_causal_mask(seq_len).to(device)

        # Get predictions
        with torch.no_grad():
            output = model(input_tensor, causal_mask)

        # Get the next token prediction (last token in the sequence)
        next_token_logits = output[0, -1, :] / temperature

        # Apply softmax to get probabilities
        probabilities = F.softmax(next_token_logits, dim=-1)

        # Sample from the probability distribution
        next_token = torch.multinomial(probabilities, 1).item()

        # Append the new token to the input sequence
        input_tensor = torch.cat([
            input_tensor,
            torch.tensor([[next_token]], dtype=torch.long).to(device)
        ], dim=1)

    # Decode and return the generated text
    generated_tokens = input_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text

# Step 14: Example usage
def main():
    # Sample data (for demonstration)
    texts = [
        "Hello, how are you doing today?",
        "Transformers are powerful neural network architectures.",
        "Language models can generate coherent text.",
        "PyTorch is a popular deep learning framework."
    ]

    # Hyperparameters
    seq_length = 20
    batch_size = 4
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 2
    dropout = 0.1
    num_epochs = 10
    learning_rate = 0.001

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = SimpleTokenizer(texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout
    )

    # Print model summary
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Train the model
    model = train_transformer_lm(model, dataloader, num_epochs, learning_rate, device)

    # eval
    model.eval()

    # save
    torch.save(model.state_dict(), "checkpoints.pth")

    # Generate text
    start_text = "Paris is"
    generated_text = generate_text(model, tokenizer, start_text, max_length=50, device=device)
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main()
