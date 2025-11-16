import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding (transformer style).

    Args:
        embed_dim: dimensionality of the positional vector (must be divisible by 4).
        temperature: scaling base used to build frequency spectrum.
    """
    def __init__(self, embed_dim=2048, temperature=10000.0):
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4 for 2D sinusoidal encodings.")
        self.embed_dim = embed_dim
        self.num_feats = embed_dim // 4
        self.temperature = temperature

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: tensor of shape (N, 2) with (x_j, y_j) patch-center coordinates.
        Returns:
            Tensor of shape (N, embed_dim).
        """
        if coords.dim() != 2 or coords.size(1) != 2:
            raise ValueError("coords must have shape (N, 2).")
        coords = coords.float()

        device = coords.device
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        x_embed = coords[:, 0].unsqueeze(1) / dim_t
        y_embed = coords[:, 1].unsqueeze(1) / dim_t

        pos_x = torch.stack((torch.sin(x_embed), torch.cos(x_embed)), dim=-1).flatten(1)
        pos_y = torch.stack((torch.sin(y_embed), torch.cos(y_embed)), dim=-1).flatten(1)
        return torch.cat((pos_x, pos_y), dim=1)  # (N, embed_dim)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        """
        Args:
            hidden_dim: feature dimension.
            num_heads: number of attention heads.
            ff_dim: feed-forward hidden width.
            dropout: dropout prob.
        """
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            Same shape tensor.
        """
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class GatingNet(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.gate(x)


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, 1)
        )

    def forward(self, x):
        A = self.attention(x)        # (N, 1)
        A = torch.transpose(A, 1, 0) # (1, N)
        A_softmax = F.softmax(A, dim=1)
        M = torch.mm(A_softmax, x)   # (1, L)
        return A, M


class MultiscaleMoEAMIL(nn.Module):
    def __init__(self, n_classes=1, dim_20x=1536, dim_5x=1536, hidden_dim=256,
                 num_experts_20x=4, num_experts_5x=4,
                 use_positional_encoding=False, pos_dim=2048):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.total_experts = num_experts_20x + num_experts_5x
        self.dim_20x = dim_20x
        self.dim_5x = dim_5x
        self.use_positional_encoding = use_positional_encoding

        if use_positional_encoding:
            self.positional_encoder = PositionalEncoding2D(embed_dim=pos_dim)
            self.pos_proj_20x = nn.Sequential(
                nn.Linear(pos_dim, dim_20x),
                nn.LayerNorm(dim_20x)
            )
            self.pos_proj_5x = nn.Sequential(
                nn.Linear(pos_dim, dim_5x),
                nn.LayerNorm(dim_5x)
            )

        self.experts_20x = nn.ModuleList([
            nn.ModuleList([Expert(dim_20x, hidden_dim) for _ in range(num_experts_20x)])
            for _ in range(n_classes)
        ])
        self.experts_5x = nn.ModuleList([
            nn.ModuleList([Expert(dim_5x, hidden_dim) for _ in range(num_experts_5x)])
            for _ in range(n_classes)
        ])

        self.gates = nn.ModuleList([
            GatingNet(dim_20x + dim_5x, self.total_experts)
            for _ in range(n_classes)
        ])
        self.inter = TransformerLayer(hidden_dim, 8, 128, dropout=0.1)

        self.attentions = nn.ModuleList([
            Attn_Net_Gated(L=hidden_dim) for _ in range(n_classes)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_classes)
        ])

    def forward(self, x, coords_20x, coords_5x=None):
        """
        Args:
            x: (N, dim_20x + dim_5x) concatenated features [20× | 5×].
            coords_20x: (N, 2) coordinates for each 20× patch j (and its paired 5× patch i(j)).
            coords_5x: optional (N, 2) coordinates for 5× patches if different from 20×.
        """
        if x.size(1) != self.dim_20x + self.dim_5x:
            raise ValueError(f"Expected feature dim {self.dim_20x + self.dim_5x}, got {x.size(1)}")

        x_20x = x[:, :self.dim_20x]
        x_5x = x[:, self.dim_20x:]

        if self.use_positional_encoding:
            if coords_20x is None:
                raise ValueError("coords_20x must be provided when positional encoding is enabled.")
            if coords_5x is None:
                coords_5x = coords_20x

            coords_20x = coords_20x.to(x.device)
            coords_5x = coords_5x.to(x.device)

            p_20x = self.positional_encoder(coords_20x)
            p_5x = self.positional_encoder(coords_5x)

            x_20x = x_20x + self.pos_proj_20x(p_20x)
            x_5x = x_5x + self.pos_proj_5x(p_5x)
        else:
            p_20x = p_5x = None

        gate_input = torch.cat([x_20x, x_5x], dim=1)

        logits = torch.zeros(1, self.n_classes, device=x.device)
        A_raw = torch.empty(self.n_classes, x.size(0), device=x.device)
        gate_weight_list = []
        class_embeddings = []

        for c in range(self.n_classes):
            gate_weights = self.gates[c](gate_input)  # (N, total_experts)
            gate_weight_list.append(gate_weights)

            expert_outputs = []
            for expert in self.experts_20x[c]:
                expert_outputs.append(expert(x_20x))
            for expert in self.experts_5x[c]:
                expert_outputs.append(expert(x_5x))

            expert_outputs = torch.stack(expert_outputs, dim=2)               # (N, hidden_dim, total_experts)
            fused = torch.bmm(expert_outputs, gate_weights.unsqueeze(2)).squeeze(2)

            A, M = self.attentions[c](fused)                                  # A: (1, N), M: (1, hidden_dim)
            A_raw[c] = A.squeeze(0)
            class_embeddings.append(M.squeeze(0))

        class_tokens = torch.stack(class_embeddings, dim=0).unsqueeze(0)      # (1, n_classes, hidden_dim)
        refined_tokens = self.inter(class_tokens).squeeze(0)                  # (n_classes, hidden_dim)

        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](refined_tokens[c].unsqueeze(0))

        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).int()

        results_dict = {
            'class_tokens': refined_tokens,
            'pre_transformer_tokens': torch.stack(class_embeddings, dim=0),
            'positional_embeddings_20x': p_20x,
            'positional_embeddings_5x': p_5x
        }

        return logits, Y_prob, Y_hat, A_raw, results_dict, gate_weight_list


if __name__ == '__main__':
    model = MultiscaleMoEAMIL(n_classes=3, )
    model.load_state_dict(torch.load('/data3/ceiling/workspace/DLBCL2/save_weights/s_0_checkpoint.pt'))
    N = 1024
    inputs = torch.randn(N, 3072)
    coords = torch.rand(N, 2)  # normalized (x_j, y_j)
    logits, Y_prob, Y_hat, A_raw, results_dict, gate_weight_list = model(inputs, coords_20x=coords)
    print(Y_hat)