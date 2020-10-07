import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, n_token, d_embed, n_feature, d_feature):
        super().__init__()
        self.n_feature = n_feature
        self.d_feature = d_feature
        if n_feature > 0 and d_feature > n_feature:
            self.proj = nn.Linear(n_feature, d_feature)

        self.embed = nn.Embedding(n_token, d_embed)

    def forward(self, tokens, features):
        assert (self.n_feature == 0) or (features.size(2) == self.n_feature), "Number of features do not match"
        if tokens.shape[-1] == 1:
            tokens = tokens.squeeze()
        token_embedding = self.embed(tokens)
        if self.n_feature > 0:
            print(self.proj)
            encoding = self.proj(features) if self.d_feature > self.n_feature else features
            return torch.cat([token_embedding, encoding], dim=2)
        return token_embedding


if __name__ == '__main__':
    bsz = 5
    seq_len = 12
    n_toks = 3
    n_feature = 19
    d_feature = 100
    d_embed = 100

    tokens = torch.LongTensor(bsz, seq_len, 1).random_(1, n_toks)
    features = torch.FloatTensor(bsz, seq_len, n_feature).random_()

    emb = Embedding(n_toks, d_embed, n_feature, d_feature)
    embed_x = emb(tokens, features)
    print(embed_x.shape)
    assert embed_x.size() == (bsz, seq_len, d_embed + d_feature)

