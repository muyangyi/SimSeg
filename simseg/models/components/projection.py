import torch.nn as nn

class ComplexProjection(nn.Module):
    def __init__(
        self,
        cfg,
        embedding_dim,
        projection_dim
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.dropout = cfg.model.projection.complex_projection.drop_out
        self.projection = nn.Linear(embedding_dim, self.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.use_gpo = cfg.model.use_gpo
        self.layer_norm = nn.LayerNorm(self.projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class SimpleProjection(nn.Module):
    def __init__(
        self,
        cfg,
        embedding_dim,
        projection_dim,
        trainable=True
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.linear = nn.Linear(embedding_dim, self.projection_dim, bias=False)

        if not trainable:
            for p in self.linear.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        return self.linear(x)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x