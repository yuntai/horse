## Copyright (c) 2017 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.

#TODO use GELU?

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.nn.utils import weight_norm
import sys
#from embeddings import Embedding

class ResidualBlock(nn.Module):
    def __init__(self, d_in, d_out, groups=1, dropout=0.0):
        super().__init__()
        assert d_in % groups == 0, "Input dimension must be a multiple of groups"
        assert d_out % groups == 0, "Output dimension must be a multiple of groups"
        self.d_in = d_in
        self.d_out = d_out
        self.proj = nn.Sequential(nn.Conv1d(d_in, d_out, kernel_size=1, groups=groups),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(dropout),
                                  nn.Conv1d(d_out, d_out, kernel_size=1, groups=groups),
                                  nn.Dropout(dropout))
        if d_in != d_out:
            self.downsample = nn.Conv1d(d_in, d_out, kernel_size=1, groups=groups)

    def forward(self, x):
        assert x.size(1) == self.d_in, "x dimension does not agree with d_in"
        return x + self.proj(x) if self.d_in == self.d_out else self.downsample(x) + self.proj(x)

class GraphLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_head, dropout=0.0, attn_dropout=0.0, wnorm=True, lev=0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.lev = lev

        # To produce the query-key-value for the self-attention computation
        self.qkv_net = nn.Linear(d_model, 3*d_model)
        self.o_net = nn.Linear(n_head*d_head, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.gamma = nn.Parameter(torch.ones(4, 4))   # For different sub-matrices of D
        self.sqrtd = np.sqrt(d_head)

        if wnorm:
            self.wnorm()

    def wnorm(self):
        self.qkv_net = weight_norm(self.qkv_net, name="weight")
        self.o_net = weight_norm(self.o_net, name="weight")
        self.proj1 = weight_norm(self.proj1, name="weight")
        self.proj2 = weight_norm(self.proj2, name="weight")

    def forward(self, Z, new_mask, mask, store=False):
        bsz, n_elem, nhid = Z.size()
        n_head, d_head, d_model = self.n_head, self.d_head, self.d_model
        assert nhid == d_model, "Hidden dimension of Z does not agree with d_model"

        # Self-attention
        inp = Z
        Z = self.norm1(Z)
        V, Q, K = self.qkv_net(Z).view(bsz, n_elem, n_head, 3*d_head).chunk(3, dim=3)     # "V, Q, K"
        W = torch.einsum('bnij, bmij->binm', Q, K).type(Z.dtype) / self.sqrtd + new_mask[:,None]
        W = self.attn_dropout(F.softmax(W, dim=3).type(mask.dtype) * mask[:, None])
        if store:
            pickle.dump(W.cpu().detach().numpy(), open(f'analysis/layer_{self.lev}_W.pkl', 'wb'))
        attn_out = torch.einsum('binm,bmij->bnij', W.float(), V.type(W.dtype)).contiguous().view(bsz, n_elem, d_model)
        attn_out = self.dropout(self.o_net(F.leaky_relu(attn_out)))
        Z = attn_out + inp

        # Position-wise feed-forward
        inp = Z
        Z = self.norm2(Z)
        return self.proj2(self.dropout(F.relu(self.proj1(Z)))) + inp

class RankingHead(nn.Module):
    def __init__(self, dim, final_dim, dropout):
        super().__init__()
        self.final_lin1 = nn.Conv1d(dim, final_dim, kernel_size=1)
        self.final_res = nn.Sequential(
                             ResidualBlock(final_dim, final_dim, dropout=dropout),
                             nn.Conv1d(final_dim, 1, kernel_size=1)
                         )

    def forward(self, Z):
        Z = self.final_lin1(Z.transpose(1,2))
        res = self.final_res(Z).squeeze()
        return res
        #return torch.sigmoid(res)

class GraphTransformer(nn.Module):
    def __init__(self,
                 dim, # model dim
                 n_layers,
                 d_inner,
                 d_embed,
                 n_feat=19,
                 final_dim=280,
                 dropout=0.0,
                 dropatt=0.0,
                 final_dropout=0.0,
                 n_head=10,
                 wnorm=True,
                 n_toks=[],
                 ):
        super().__init__()

        #TODO: better way to mix tok/features?
        embeddings = []
        for i, n in enumerate(n_toks):
            embeddings.append(nn.Embedding(n+1, d_embed))

        feat_dim = dim - d_embed * len(embeddings)
        self.feat_proj = nn.Linear(n_feat, feat_dim)

        self.embeddings = embeddings
        self.dim = dim
        self.wnorm = wnorm
        self.n_head = n_head

        assert dim % n_head == 0, "dim must be a multiple of n_head"
        self.layers = nn.ModuleList([GraphLayer(d_model=dim, d_inner=d_inner, n_head=n_head, d_head=dim//n_head, dropout=dropout,
                                                attn_dropout=dropatt, wnorm=wnorm, lev=i+1) for i in range(n_layers)])
        self.final_norm = nn.LayerNorm(dim)

        self.final_dropout = final_dropout
        self.final_lin1 = nn.Conv1d(dim, final_dim, kernel_size=1)
        self.final_res = nn.Sequential(
                             ResidualBlock(final_dim, final_dim, dropout=final_dropout),
                             nn.Conv1d(final_dim, 1, kernel_size=1)
                         )
        self.n_toks = n_toks
        self.ranking_head = RankingHead(dim, final_dim, final_dropout)

        self.apply(self.weights_init)

    def forward(self, x_feats, x_toks, mask):
        mask = torch.einsum('bi,bj->bij', mask, mask)
        new_mask = -1e20 * torch.ones_like(mask).to(mask.device) # additive mask
        new_mask[mask > 0] = 0

        # create embeddings
        zs = []
        for i, n in enumerate(self.n_toks):
            zs.append(self.embeddings[i](x_toks[..., i]))

        zs.append(self.feat_proj(x_feats))

        Z = torch.cat(zs, dim=-1)

        for i in range(len(self.layers)):
            Z = self.layers[i](Z, new_mask, mask, store=False)

        Z = self.final_norm(Z) # bsz x seq x hid
        return self.ranking_head(Z)

    @staticmethod
    def init_weight(weight):
        nn.init.uniform_(weight, -0.1, 0.1)

    @staticmethod
    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1 or classname.find('Conv1d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                GraphTransformer.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                GraphTransformer.init_bias(m.bias)

