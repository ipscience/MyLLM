# gpt_projectフォルダ内に以下のサブフォルダを作成します：
# 
# data: データセットを保存
# models: チェックポイントと最終モデルを保存
# logs: トレーニングログを保存
# 
# Google Colab Pro+でバックグラウンド処理を行うことを想定して、
# 
# データセット: My Drive/gpt_project/data/input.txt
# チェックポイント: My Drive/gpt_project/models/gpt_checkpoint.pt
# 最終モデル: My Drive/gpt_project/models/gpt_model_final.pth
# トレーニングログ: My Drive/gpt_project/logs/training_log.txt

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import Counter
import requests
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download dataset if not already in Google Drive
drive_path = "/content/drive/MyDrive/gpt_project"
os.makedirs(drive_path, exist_ok=True)
dataset_path = f"{drive_path}/input.txt"

if not os.path.exists(dataset_path):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Dataset downloaded and saved to Google Drive")
else:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print("Dataset loaded from Google Drive")

# Hyperparameters
batch_size = 64
block_size = 128
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# Tokenize text
words = text.split()

# Create vocabulary
word_counts = Counter(words)
vocab = ['<UNK>'] + [word for word, count in word_counts.items() if count > 1]
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")

# Word to index and index to word mappings
stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for i, word in enumerate(vocab)}

# Encoder and decoder functions
encode = lambda s: [stoi.get(word, 0) for word in s.split()]
decode = lambda l: ' '.join([itos[i] for i in l])

# Convert text to tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Train and validation splits
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Model architecture (Head, MultiHeadAttention, FeedForward, Block, GPTLanguageModel)
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # focus only on the last time step
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Instantiate the model
model = GPTLanguageModel().to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Load checkpoint if exists
checkpoint_path = f"{drive_path}/gpt_checkpoint.pt"
start_iter = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iteration']
    print(f"Resuming from iteration {start_iter}")

# Training loop
for iter in range(start_iter, max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save checkpoint
        torch.save({
            'iteration': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses['train'],
        }, checkpoint_path)
        print(f"Checkpoint saved at iteration {iter}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_new_tokens=100)[0].tolist()

generated_words = decode(generated_tokens)
print("\nGenerated text:")
print(generated_words)

# Save the final model
final_model_path = f"{drive_path}/gpt_model_final.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")