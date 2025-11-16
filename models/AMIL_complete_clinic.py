import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding (Transformer style).
    embed_dim 必须能被 4 整除。
    """
    def __init__(self, embed_dim=512, temperature=10000.0):
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim 必须能被 4 整除用于 2D 位置编码。")
        self.embed_dim = embed_dim
        self.num_feats = embed_dim // 4
        self.temperature = temperature

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (N, 2) 的 (x, y) 坐标张量。
        return: (N, embed_dim) 的位置向量。
        """
        if coords.dim() != 2 or coords.size(1) != 2:
            raise ValueError("coords 张量形状必须为 (N, 2)。")
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
        super(TransformerLayer, self).__init__()
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
        A = self.attention(x)   # (N, 1)
        A = torch.transpose(A, 1, 0)  # (1, N)
        A_softmax = F.softmax(A, dim=1)
        M = torch.mm(A_softmax, x)    # (1, L)
        return A, M


class MultiscaleMoEAMIL(nn.Module):
    def __init__(self, n_classes=1, dim_20x=1536, dim_5x=1536, hidden_dim=256,
                 num_experts_20x=4, num_experts_5x=4,
                 use_positional_encoding=True, pos_dim=512, temperature=10000.0):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.total_experts = num_experts_20x + num_experts_5x
        self.dim_20x = dim_20x
        self.dim_5x = dim_5x
        self.use_positional_encoding = use_positional_encoding

        if use_positional_encoding:
            self.positional_encoder = PositionalEncoding2D(embed_dim=pos_dim, temperature=temperature)
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

    def forward(self, x, coords_20x=None, coords_5x=None):
        """
        x: (N, dim_20x + dim_5x)
        coords_20x / coords_5x: (N, 2) 2D 坐标（若仅传 coords_20x，则 5× 使用相同坐标）
        """
        if x.size(1) != self.dim_20x + self.dim_5x:
            raise ValueError(f"输入特征维度应为 {self.dim_20x + self.dim_5x}，当前为 {x.size(1)}。")

        x_20x = x[:, :self.dim_20x]
        x_5x = x[:, self.dim_20x:]
        pos20 = pos5 = None

        if self.use_positional_encoding:
            if coords_20x is None:
                raise ValueError("启用位置编码时必须提供 coords_20x。")
            if coords_5x is None:
                coords_5x = coords_20x

            coords_20x = coords_20x.to(x.device)
            coords_5x = coords_5x.to(x.device)

            pos20 = self.positional_encoder(coords_20x)
            pos5 = self.positional_encoder(coords_5x)

            x_20x = x_20x + self.pos_proj_20x(pos20)
            x_5x = x_5x + self.pos_proj_5x(pos5)

        gate_input = torch.cat([x_20x, x_5x], dim=1)

        logits = torch.zeros(1, self.n_classes, device=x.device)
        A_raw = torch.empty(self.n_classes, x.size(0), device=x.device)
        gate_weight_list = []
        features = torch.empty(self.n_classes, self.hidden_dim, device=x.device)

        for c in range(self.n_classes):
            gate_weights = self.gates[c](gate_input)  # (N, total_experts)
            gate_weight_list.append(gate_weights)

            expert_outputs = []
            for expert in self.experts_20x[c]:
                expert_outputs.append(expert(x_20x))
            for expert in self.experts_5x[c]:
                expert_outputs.append(expert(x_5x))

            expert_outputs = torch.stack(expert_outputs, dim=2)  # (N, hidden_dim, total_experts)
            fused = torch.bmm(expert_outputs, gate_weights.unsqueeze(2)).squeeze(2)  # (N, hidden_dim)

            A, M = self.attentions[c](fused)
            A_raw[c] = A.squeeze(0)
            features[c] = M.squeeze(0)

        features = self.inter(features.unsqueeze(0)).squeeze(0)

        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](features[c])

        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).int()
        results_dict = {
            'refined_features': features,
            'attention_maps': A_raw,
            'positional_embeddings_20x': pos20,
            'positional_embeddings_5x': pos5
        }

        return features, logits, Y_prob, Y_hat, A_raw, results_dict, gate_weight_list


class ModelWithClinic(nn.Module):
    def __init__(self,
                 n_clinic: int = 13,
                 n_classes: int = 3,
                 img_dim: int = 256,
                 clinic_dim: int = 64,
                 hidden_dim: int = 128,
                 dropout: float = 0.5):
        super().__init__()
        self.n_classes = n_classes
        self.img_dim = img_dim

        self.clinic_encoder = nn.Sequential(
            nn.Linear(n_clinic, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, clinic_dim),
            nn.ReLU(inplace=True)
        )

        self.integrators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(img_dim + clinic_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ) for _ in range(n_classes)
        ])

        self.class_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_classes)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        """
        x = (image_feature, clinic_info)
            image_feature : (C, img_dim)
            clinic_info   : (13,)
        """
        image_feature, clinic_info = x

        if clinic_info.ndim == 1:
            clinic_info = clinic_info.unsqueeze(0)

        clinic_feat = self.clinic_encoder(clinic_info)        # (1, clinic_dim)
        clinic_feat = clinic_feat.expand(self.n_classes, -1)  # (C, clinic_dim)

        logits_list = []
        fuse_feat_lst = []
        for c in range(self.n_classes):
            fusion = torch.cat([image_feature[c], clinic_feat[c]], dim=0)
            fusion = self.integrators[c](fusion)
            logit = self.class_heads[c](fusion)
            logits_list.append(logit)
            fuse_feat_lst.append(fusion)

        logits = torch.stack(logits_list, dim=1)
        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).int()
        features = torch.stack(fuse_feat_lst)

        results_dict = {'features': features}

        return logits, Y_prob, Y_hat, None, results_dict, None

# ──────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────
if __name__ == "__main__":
    # Multiscale model demo (输出图像特征 + 位置编码)
    bag_size = 1024
    multi_model = MultiscaleMoEAMIL(n_classes=3, hidden_dim=256, use_positional_encoding=True)
    bag_feats = torch.randn(bag_size, 3072)
    coords = torch.rand(bag_size, 2)  # 归一化坐标
    img_features, _, _, _, _, _, _ = multi_model(bag_feats, coords_20x=coords)

    # Clinic fusion demo
    clinic_in = torch.randn(13)
    fusion_model = ModelWithClinic(n_classes=3, img_dim=256)
    logits, probs, hats, _, res, _ = fusion_model((img_features, clinic_in))

    print("logits :", logits)
    print("probs  :", probs)
    print("hats   :", hats)
    print("fused feature shape :", res['features'].shape)