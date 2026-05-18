import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class HeadwiseDynamicTemperature(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temp_predictor = nn.Sequential(
            nn.Linear(d_model, num_heads),
            nn.Softplus()
        )
        nn.init.constant_(self.temp_predictor[0].bias, 2.0)
    
    def forward(self, context):
        sent_repr = context.mean(dim=1)  # [batch, d_model]
        temps = self.temp_predictor(sent_repr)  # [batch, num_heads]
        return temps.unsqueeze(-1).unsqueeze(-1)  # [batch, num_heads, 1, 1]


def scaled_dot_product_attention_standard(Q, K, V, mask=None, dropout=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights, V), attn_weights

def scaled_dot_product_attention_hdts(Q, K, V, temperature, mask=None, dropout=None):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / temperature  # 动态缩放
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights, V), attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_output, attn_weights = scaled_dot_product_attention_standard(Q, K, V, mask, self.dropout)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_O(attn_output), attn_weights

class MultiHeadAttention_HDTS(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.hdts = HeadwiseDynamicTemperature(d_model, num_heads)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        temperature = self.hdts(query)  # [batch, num_heads, 1, 1]
        attn_output, attn_weights = scaled_dot_product_attention_hdts(Q, K, V, temperature, mask, self.dropout)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_O(attn_output), attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_hdts=False):
        super().__init__()
        Attn = MultiHeadAttention_HDTS if use_hdts else MultiHeadAttention
        self.self_attn = Attn(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_hdts=False):
        super().__init__()
        Attn = MultiHeadAttention_HDTS if use_hdts else MultiHeadAttention
        self.masked_self_attn = Attn(d_model, num_heads, dropout)
        self.enc_dec_attn = Attn(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_out, _ = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        attn_out, _ = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_len=100, dropout=0.1, use_hdts=False):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, use_hdts)
            for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, use_hdts)
            for _ in range(num_decoder_layers)])
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask=None):
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.output_layer(dec_output)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CopyTaskDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=32):
        self.data = torch.randint(3, vocab_size, (num_samples, seq_len))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        src = self.data[idx]
        tgt_in = torch.cat([torch.tensor([1]), src])   # <sos>
        tgt_out = torch.cat([src, torch.tensor([2])])  # <eos>
        return src, tgt_in, tgt_out


def create_masks(src, tgt_input, pad_idx=0):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_pad_mask = (tgt_input != pad_idx).unsqueeze(1).unsqueeze(3)
    tgt_len = tgt_input.size(1)
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt_input.device))
    tgt_mask = tgt_pad_mask & causal_mask.unsqueeze(0).unsqueeze(0)
    return src_mask, tgt_mask


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_tokens = 0, 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        src_mask, tgt_mask = create_masks(src, tgt_in)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        optimizer.zero_grad()
        output = model(src, tgt_in, src_mask, tgt_mask)
        output = output.view(-1, output.size(-1))
        tgt_out = tgt_out.view(-1)
        loss = criterion(output, tgt_out)
        loss.backward()
        optimizer.step()
        mask = (tgt_out != 0)
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()
    return total_loss / total_tokens


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens, total_correct, total_preds = 0, 0, 0, 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_mask, tgt_mask = create_masks(src, tgt_in)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
            output = model(src, tgt_in, src_mask, tgt_mask)
            output_flat = output.view(-1, output.size(-1))
            tgt_flat = tgt_out.view(-1)
            loss = criterion(output_flat, tgt_flat)
            mask = (tgt_flat != 0)
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
            preds = output.argmax(dim=-1)
            total_correct += ((preds == tgt_out) & (tgt_out != 0)).sum().item()
            total_preds += mask.sum().item()
    return total_loss / total_tokens, total_correct / total_preds


if __name__ == "__main__":
    VOCAB_SIZE = 16
    D_MODEL = 32
    NUM_HEADS = 4
    NUM_LAYERS = 2
    D_FF = 128
    SEQ_LEN = 8
    BATCH_SIZE = 16
    NUM_EPOCHS = 25
    LR = 1e-3

    train_ds = CopyTaskDataset(500, SEQ_LEN, VOCAB_SIZE)
    val_ds = CopyTaskDataset(100, SEQ_LEN, VOCAB_SIZE)
    test_ds = CopyTaskDataset(100, SEQ_LEN, VOCAB_SIZE)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    model_std = Transformer(VOCAB_SIZE, VOCAB_SIZE, D_MODEL, NUM_HEADS,
                            NUM_LAYERS, NUM_LAYERS, D_FF, SEQ_LEN+5, 0.1, False).to(DEVICE)
    model_hdts = Transformer(VOCAB_SIZE, VOCAB_SIZE, D_MODEL, NUM_HEADS,
                             NUM_LAYERS, NUM_LAYERS, D_FF, SEQ_LEN+5, 0.1, True).to(DEVICE)
    
    print(f"M1 (标准) 参数量: {model_std.count_parameters():,}")
    print(f"M2 (HDTS) 参数量: {model_hdts.count_parameters():,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def run_training(model, epochs, lr, name):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if (epoch + 1) % 5 == 0:
                print(f"[{name}] Epoch {epoch+1}/{epochs} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}")
        return history
    
    print("\n训练 M1 (标准Transformer)...")
    history_m1 = run_training(model_std, NUM_EPOCHS, LR, "M1标准")
    
    print("\n训练 M2 (HDTS Transformer)...")
    history_m2 = run_training(model_hdts, NUM_EPOCHS, LR, "M2_HDTS")

    test_loss_m1, test_acc_m1 = evaluate(model_std, test_loader, criterion, DEVICE)
    test_loss_m2, test_acc_m2 = evaluate(model_hdts, test_loader, criterion, DEVICE)
    
    print("\n" + "="*60)
    print("测试集最终结果:")
    print(f"M1 (标准)  Loss: {test_loss_m1:.4f}  Acc: {test_acc_m1:.4f}")
    print(f"M2 (HDTS)  Loss: {test_loss_m2:.4f}  Acc: {test_acc_m2:.4f}")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history_m1['train_loss'], 'b-', label='M1 Standard')
    axes[0].plot(history_m2['train_loss'], 'r--', label='M2 HDTS')
    axes[0].set_title('Training Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history_m1['val_loss'], 'b-', label='M1 Standard')
    axes[1].plot(history_m2['val_loss'], 'r--', label='M2 HDTS')
    axes[1].set_title('Val Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history_m1['val_acc'], 'b-', label='M1 Standard')
    axes[2].plot(history_m2['val_acc'], 'r--', label='M2 HDTS')
    axes[2].set_title('Val Accuracy'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_results_hdts.png', dpi=150)
    print("\n图表已保存为 experiment_results_hdts.png")
