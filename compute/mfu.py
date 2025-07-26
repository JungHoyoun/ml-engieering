import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# FFN 구현 (GELU vs SwiGLU)
# ----------------------------
class FFN_GELU(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn, bias=False)
        self.fc2 = nn.Linear(d_ffn, d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        hidden = int(2*d_ffn/3)  # GLU variant 논문 기준
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# ----------------------------
# Attention 구현 (scaled dot-product + projection)
# ----------------------------
class SimpleAttention(nn.Module):
    def __init__(self, d_model, heads, k_per_head=None):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)
        self.k_per_head = k_per_head or heads

    def forward(self, x):
        B, N, C = x.size()
        q = self.to_q(x).view(B, N, self.heads, self.d_k)
        k = self.to_k(x).view(B, N, self.k_per_head, self.d_k)
        v = self.to_v(x).view(B, N, self.k_per_head, self.d_k)
        # FLOPs counting 지표로 단순화: dot‑prod N^2 * d_k * heads 등 생략
        return self.to_out(q @ k.transpose(-2,-1) @ v.reshape(B,N,self.k_per_head * self.d_k))

# ----------------------------
# FLOPs 계산 함수
# ----------------------------
def count_flops_ffn(d_model, d_ffn, ffn_type):
    if ffn_type == 'GELU':
        return d_model * d_ffn + d_model * d_ffn  # 두 개의 Linear
    elif ffn_type == 'SwiGLU':
        hidden = int(2 * d_ffn / 3)
        # w1, w2, elementwise, w3
        return d_model*hidden + d_model*hidden + hidden + hidden + hidden*d_model
    else:
        raise ValueError()

def count_flops_attn(d_model, heads, k_per_head):
    # 단순히 Q,K,V proj + output proj
    return 3 * (d_model * d_model) + d_model * d_model

# ----------------------------
# 구성 & 비교
# ----------------------------
configs = [
    {'ffn': 'SwiGLU', 'r': 8/3, 'attn_heads': 32, 'k_per_head': 8},
    {'ffn': 'SwiGLU', 'r': 8/3, 'attn_heads': 32, 'k_per_head': 32},
    {'ffn': 'SwiGLU', 'r': 4,     'attn_heads': 32, 'k_per_head': 32},
    {'ffn': 'GELU',   'r': 4,     'attn_heads': 32, 'k_per_head': 32},
    {'ffn': 'GELU',   'r': 1,     'attn_heads': 32, 'k_per_head': 32},
]
D = 4096  # 예시 d_model
for cfg in configs:
    d_ffn = int(cfg['r'] * D)
    flops_ffn = count_flops_ffn(D, d_ffn, cfg['ffn'])
    flops_attn = count_flops_attn(D, cfg['attn_heads'], cfg['k_per_head'])
    print(cfg, '→ total TFLOPs ≈', (flops_ffn + flops_attn)/1e9)
