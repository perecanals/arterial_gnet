import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head_dim, num_heads, use_bias=True):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scaling = self.head_dim ** -0.5

        self.Q = nn.Linear(dim, head_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(dim, head_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(dim, head_dim * num_heads, bias=use_bias)
        self.O = nn.Linear(head_dim * num_heads, dim, bias=use_bias)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)
        
        q = self.Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            # Expand mask to match attention shape
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            mask = mask.expand(-1, self.num_heads, seq_len, -1)  # [batch_size, num_heads, seq_len, seq_len]
            attention = attention.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.O(out)

class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, use_bias=True, layer_norm=True):
        super().__init__()
        self.attention = MultiHeadAttention(dim, dim // num_heads, num_heads, use_bias)
        self.norm1 = nn.LayerNorm(dim) if layer_norm else nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.FFN_layer1 = nn.Linear(dim, dim * 2)
        self.FFN_layer2 = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim) if layer_norm else nn.BatchNorm1d(dim)

    def forward(self, x, mask=None):
        attended = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attended))
        ffn_output = self.FFN_layer1(x)
        ffn_output = F.relu(ffn_output)
        ffn_output = self.FFN_layer2(ffn_output)
        return self.norm2(x + self.dropout(ffn_output))
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_out_layers = 2):
        super().__init__()
        list_linear_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(num_out_layers - 1)]
        list_linear_layers.append(nn.Linear(input_dim // 2 ** (num_out_layers - 1), output_dim, bias=True))
        self.linear_layers = nn.ModuleList(list_linear_layers)
        self.num_out_layers = num_out_layers
        
    def forward(self, x):
        for l in range(self.num_out_layers - 1):
            x = self.linear_layers[l](x)
            x = F.relu(x)
        return self.linear_layers[self.num_out_layers - 1](x)

def sequence_mean_pool(x, seq_len, mask=None):
    # Sum the values and divide by sequence length
    return x.sum(dim=1) / seq_len.unsqueeze(1).float()

def sequence_max_pool(x, seq_len, mask=None):
    # Set padding to negative infinity
    x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
    # Take the max along the sequence dimension
    return x_masked.max(dim=1)[0]

def sequence_sum_pool(x, seq_len, mask=None):
    # Sum the values, automatically ignoring padding
    return (x * mask.unsqueeze(-1)).sum(dim=1)

class TransformerNet(nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim,
            out_dim,
            num_layers=1,
            attn_heads=8,
            dropout=0.1,
            use_bias=True,
            layer_norm=True,
            aggregation="mean",
            num_out_layers=1,
            pos_enc_dim=8
    ):
        super().__init__()
        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, attn_heads, dropout, use_bias, layer_norm)
            for _ in range(num_layers)
        ])
        
        if aggregation == "mean":
            self.aggregation = sequence_mean_pool
        elif aggregation == "max":
            self.aggregation = sequence_max_pool
        elif aggregation == "sum":
            self.aggregation = sequence_sum_pool
        
        self.out_mlp = MLP(hidden_dim, out_dim, num_out_layers)

    def forward(self, batch):
        x, mask, pos_enc, seq_len = batch.x, batch.mask, batch.pos_enc, batch.seq_len
        
        x = self.embedding(x)
        x = x + self.embedding_pos_enc(pos_enc)
        
        # Create attention mask
        mask = mask.squeeze(1)  # Remove the extra dimension if present
        
        for layer in self.layers:
            x = layer(x, mask)
        
        # Use the sequence-aware aggregation function
        x = self.aggregation(x, seq_len, mask)
        
        x = self.out_mlp(x)
        
        x = F.softmax(x, dim=1)

        return x.unsqueeze(0)
