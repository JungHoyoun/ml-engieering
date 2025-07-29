Parameter 계산법

Dense Model
```
# GPT-2 parameter count calculator

# Define constants for GPT-2 small model
V = 50257  # Vocabulary size
E = 768    # Embedding size
P = 1024   # Max sequence length
L = 12     # Number of transformer layers
H = 3072   # Hidden layer size (MLP)

# Compute total parameter count using the formula:
# C = E(V + P) + L(12E^2 + 13E) + 2E

def compute_gpt2_parameters(V, E, P, L):
    term1 = E * (V + P)
    term2 = L * (12 * E**2 + 13 * E)
    term3 = 2 * E
    total_params = term1 + term2 + term3
    return total_params

total_params = compute_gpt2_parameters(V, E, P, L)
print(f"Total GPT-2 parameter count: {total_params:,}")

```

MoE Model
```

```