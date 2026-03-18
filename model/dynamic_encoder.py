import torch
import torch.nn as nn


class DynamicStateEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_states = getattr(args, "dyn_states", 4)
        self.embed_dim = getattr(args, "dyn_embed_dim", 32)
        self.hidden_dim = getattr(args, "dyn_hidden_dim", 64)
        self.out_dim = args.node_dim
        self.padding_idx = self.num_states

        self.embedding = nn.Embedding(
            num_embeddings=self.num_states + 1,
            embedding_dim=self.embed_dim,
            padding_idx=self.padding_idx,
        )
        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.out_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )

    def forward(self, dyn_seq):
        if dyn_seq is None:
            raise ValueError("dyn_seq cannot be None when using DynamicStateEncoder.")

        if not torch.is_tensor(dyn_seq):
            dyn_seq = torch.as_tensor(
                dyn_seq, dtype=torch.long, device=self.embedding.weight.device
            )
        else:
            dyn_seq = dyn_seq.to(self.embedding.weight.device).long()

        mask = (dyn_seq != self.padding_idx).unsqueeze(-1).float()
        emb = self.embedding(dyn_seq)
        out, _ = self.gru(emb)

        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        dyn_embed = self.proj(pooled)
        return dyn_embed