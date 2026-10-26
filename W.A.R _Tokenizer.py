import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tiktoken
  
enc = tiktoken.get_encoding("gpt2")   # vocab 50257, BPE pronto
  
def encode(text):
    return enc.encode(text)             # texto → lista de ints
  
def decode(ids):
    return enc.decode(ids)              # lista de ints → texto
vocab_size = 50257  # tiktoken gpt2 (50256 BPE + 1 especial)
d_model = 768
context_length = 1000
n_heads = 12
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, debug=False):
        super().__init__()
        self.debug = debug
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B = x.shape[0]
        T = x.shape[1]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = Q.reshape(B, T, self.n_heads, self.d_head)
        Q = Q.transpose(1,2)
        K = K.reshape(B, T, self.n_heads, self.d_head)
        K = K.transpose(1,2)
        K = K.transpose(2,3)
        V = V.reshape(B, T, self.n_heads, self.d_head)
        V = V.transpose(1,2)
        scores = Q @ K
        scores = scores/(self.d_head ** 0.5)
        mask = torch.triu(torch.ones(T, T), diagonal=1) * -1e9
        scores += mask
        pesos = torch.softmax(scores, dim=-1)
        saida = pesos @ V
        saida = saida.transpose(1, 2)
        saida = saida.contiguous()             
        saida = saida.reshape(B, T, d_model)   
        saida = self.out_proj(saida)
        if self.debug:
            print(saida.shape)
            print(x.shape)
        return saida
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out = self.mha(x_norm)
        x = x + attn_out  
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out  
        
        return x

n_blocks = 12

class GPT(nn.Module):
    def __init__(self, d_model, n_heads, n_blocks, vocab_size, context_length):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_blocks)]
        )
        # weight tying: lm_head e token_emb compartilham pesos
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens):
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


gpt = GPT(d_model, n_heads, n_blocks, vocab_size, context_length)

# Carregar TinyStories já tokenizado

train_tokens = torch.load("tinystories_train.pt")
val_tokens = torch.load("tinystories_val.pt")

print("Train:", train_tokens.shape)
print("Validation:", val_tokens.shape)

optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)


losses = []


# Pega uma sequência do TinyStories
tokens = train_tokens[:context_length + 1]
# =========================
# TREINAMENTO
# =========================

context_length = 256
batch_size = 1
epochs = 1

optimizer = torch.optim.AdamW(
    gpt.parameters(),
    lr=1e-3
)

losses = []

gpt.train()

for epoch in range(epochs):

    print(f"\n===== EPOCH {epoch + 1}/{epochs} =====")

    # percorre TODO o dataset
    for start in range(
        0,
        len(train_tokens) - context_length - 1,
        context_length
    ):

        # pega um bloco de 1001 tokens
        tokens = train_tokens[
            start:start + context_length + 1
        ]

        # adiciona dimensão do batch
        tokens = tokens.unsqueeze(0)

        # entrada e alvo
        inputs = tokens[:, :-1].long()
        targets = tokens[:, 1:].long()

        # forward
        logits = gpt(inputs)

        # loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # progresso
        if start % (context_length * 100) == 0:

            progresso = (
                start / len(train_tokens)
            ) * 100

            print(
                f"Tokens: {start:,}/{len(train_tokens):,} "
                f"({progresso:.2f}%) | "
                f"Loss: {loss.item():.4f}"
            )

print("\nTreinamento terminado!")

# Gráfico da loss
plt.plot(losses)

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.grid(True)
plt.show()

gpt.eval()
with torch.no_grad():

    while True:
        texto = input("talk with WAR:   ")

        ids = enc.encode(texto)

        tokens = torch.tensor(
            [ids],
            dtype=torch.long
        )

        for _ in range(50):

            logits = gpt(tokens)

            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

        print(enc.decode(tokens[0].tolist()))