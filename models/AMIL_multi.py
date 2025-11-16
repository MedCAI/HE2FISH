import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        """
        初始化 Transformer Layer。
        :param hidden_dim: 输入和输出的维度 (hidden layer size)。
        :param num_heads: 多头注意力的头数。
        :param ff_dim: 前馈网络中间层的维度。
        :param dropout: Dropout 比例。
        """
        super(TransformerLayer, self).__init__()
        
        # 多头注意力机制
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        
        # Layer Normalization 和 Dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为 (N, hidden_dim)，其中 N 是输入序列的长度。
        :return: 输出张量，形状为 (N, hidden_dim)。
        """
        # Self-Attention
        attn_output, _ = self.self_attention(x, x, x)  # Query=Key=Value=x
        x = self.norm1(x + self.dropout1(attn_output))  # 残差连接 + LayerNorm
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))  # 残差连接 + LayerNorm
        
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
        A = self.attention(x)  # (N, 1)
        A = torch.transpose(A, 1, 0)  # (1, N)
        A_softmax = F.softmax(A, dim=1)
        M = torch.mm(A_softmax, x)  # (1, L)
        return A, M


class MultiscaleMoEAMIL(nn.Module):
    def __init__(self, n_classes=1, dim_20x=1536, dim_5x=1536, hidden_dim=256,
                 num_experts_20x=4, num_experts_5x=4):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.total_experts = num_experts_20x + num_experts_5x

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
        

    def forward(self, x):
        x_20x, x_5x = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]
        logits = torch.zeros(1, self.n_classes, device=x.device)
        A_raw = torch.empty(self.n_classes, x.size(0), device=x.device)
        gate_weight_list = []
        features =  torch.empty(self.n_classes, self.hidden_dim, device=x.device)

        for c in range(self.n_classes):
            gate_weights = self.gates[c](x)  # (N, total_experts)
            gate_weight_list.append(gate_weights)

            expert_outputs = []
            for expert in self.experts_20x[c]:
                expert_outputs.append(expert(x_20x))
            for expert in self.experts_5x[c]:
                expert_outputs.append(expert(x_5x))

            expert_outputs = torch.stack(expert_outputs, dim=2)  # (N, hidden_dim, total_experts)
            fused = torch.bmm(expert_outputs, gate_weights.unsqueeze(2)).squeeze(2)  # (N, hidden_dim)

            A, M = self.attentions[c](fused)
            A_raw[c] = A
            features[c] = M
        features = self.inter(features)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M)

        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).int()
        results_dict = {'features': M}

        return logits, Y_prob, Y_hat, A_raw, results_dict, gate_weight_list    

if __name__ == '__main__':
    model = MultiscaleMoEAMIL(n_classes=3)
    inputs = torch.randn(1024, 3072)
    logits, Y_prob, Y_hat, A_raw, results_dict, gate_weight_list = model(inputs)
    print(Y_hat)
