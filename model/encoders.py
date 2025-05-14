import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data
from collections import OrderedDict
from einops import repeat, rearrange
from transformers import AutoModelWithLMHead
from mamba_ssm import Mamba2
import numpy as np


class MolEncoder(nn.Module):  # graph
    def __init__(self,
                 node_in_feats, edge_in_feats,  # 9, 3
                 hidden_size,  # 32
                 num_step_mp,  # 1
                 num_step_set2set,  # 1
                 num_layer_set2set,  # 1
                 output_dim  # 1024
                 ):
        super(MolEncoder, self).__init__()
        self.linear = nn.Sequential(nn.Linear(node_in_feats, hidden_size), nn.ReLU())
        self.num_step_mp = num_step_mp
        edge_network = nn.Sequential(nn.Linear(edge_in_feats, hidden_size * hidden_size), nn.ReLU())
        self.gnn_layer = NNConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            nn=edge_network,
            aggr='add'
        )
        self.activation = nn.ReLU()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.readout = Set2Set(in_channels=hidden_size * 2,
                               processing_steps=num_step_set2set,
                               num_layers=num_layer_set2set)

        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.PReLU(),
            nn.Dropout(0.1)
        )
        self.sparsify = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.PReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, data, seq_feat):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        node_feats = self.linear(x.float())
        edge_attr = edge_attr.float()
        hidden_feats = node_feats.unsqueeze(0)
        node_aggr = [node_feats]

        for _ in range(self.num_step_mp):
            node_feats = self.activation(self.gnn_layer(node_feats, edge_index, edge_attr)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)

        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)

        readout = self.readout(node_aggr, batch)  # (11, 256)
        if len(readout.shape) == 1:
            readout = readout.unsqueeze(0)
        n = readout.shape[0]

        # mol-fn
        graph_origin = self.proj(readout)  # (11, 1024)
        if seq_feat is not None:
            seq_feat = seq_feat.unsqueeze(0)  # (1, 1024)
            seq_feat = repeat(seq_feat, 'k c -> (m k) c', m=n)  # (11, 1024)
            residual = torch.cat([graph_origin, seq_feat], dim=-1)  # (11, 2048)
        else:
            residual = graph_origin
        fused_feats = self.sparsify(residual)
        return fused_feats  # (11, 1024)


class SMILES_Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,  # 512
                 num_heads,  # 2
                 num_layers,  # 4
                 context_length=1024,  # 512
                 model_path='./ChemBERTa-2',
                 output_dim=512,  # 1024
                 lock=True,
                 initialize=False,
                 lock_all=False
                 ):
        super(SMILES_Encoder, self).__init__()
        embed_dim = 767
        self.transformer_encoder = AutoModelWithLMHead.from_pretrained(model_path)
        if lock:
            for name, parameter in self.transformer_encoder.named_parameters():
                if 'head' not in name:
                    parameter.requires_grad = False
        if initialize:
            for param in self.transformer_encoder.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        self.ln_final = nn.LayerNorm(embed_dim)
        self.pooler = nn.Linear(embed_dim, output_dim)
        if lock_all:
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False
            for param in self.ln_final.parameters():
                param.requires_grad = False
            for param in self.pooler.parameters():
                param.requires_grad = False
            print('parameters all locked!')

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            attention_mask = ~attention_mask

        x = self.transformer_encoder(input_ids, attention_mask)[0]

        x = self.ln_final(x)
        x = self.pooler(x[:, 0, :])
        return x

class MHA(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = heads
        self.out_dim = out_dim
        self.wq = nn.Linear(in_dim, out_dim, bias=False)
        self.wk = nn.Linear(in_dim, out_dim, bias=False)
        self.wv = nn.Linear(in_dim, out_dim, bias=False)
        self.dk = 1 / math.sqrt(in_dim // heads)

    def forward(self, x):
        b, l, _ = x.shape
        Q = self.wq(x).reshape(b, l, self.num_heads, self.out_dim // self.num_heads).transpose(1, 2)
        K = self.wk(x).reshape(b, l, self.num_heads, self.out_dim // self.num_heads).transpose(1, 2)
        V = self.wv(x).reshape(b, l, self.num_heads, self.out_dim // self.num_heads).transpose(1, 2)
        dist = torch.matmul(Q, K.transpose(2, 3)) * self.dk  # (Q * K^T) / dk
        dist = torch.softmax(dist, dim=-1)
        attn = torch.matmul(dist, V)
        attn = attn.transpose(1, 2).reshape(b, l, self.out_dim)
        return attn


class crossMHA(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = heads
        self.out_dim = out_dim
        self.wq = nn.Linear(in_dim, out_dim, bias=False)
        self.wk = nn.Linear(in_dim, out_dim, bias=False)
        self.wv = nn.Linear(in_dim, out_dim, bias=False)
        self.dk = 1 / math.sqrt(in_dim // heads)

    def forward(self, x, y):
        b, l, _ = x.shape
        _, ly, _ = y.shape
        # (128, 8, 7, 64)
        Q = self.wq(x).reshape(b, l, self.num_heads, self.out_dim // self.num_heads).transpose(1, 2)
        K = self.wk(y).reshape(b, ly, self.num_heads, self.out_dim // self.num_heads).transpose(1, 2)
        V = self.wv(y).reshape(b, ly, self.num_heads, self.out_dim // self.num_heads).transpose(1, 2)
        # attention
        dist = torch.matmul(Q, K.transpose(2, 3)) * self.dk  # (Q * K^T) / dk
        dist = torch.softmax(dist, dim=-1)
        attn = torch.matmul(dist, V)
        attn = attn.transpose(1, 2).reshape(b, l, self.out_dim)
        return attn


class CrossAttnBlock(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = heads
        self.out_dim = out_dim
        self.wq = nn.Linear(in_dim, out_dim, bias=False)
        self.wk = nn.Linear(in_dim, out_dim, bias=False)
        self.wv = nn.Linear(in_dim, out_dim, bias=False)
        self.dk = 1 / math.sqrt(in_dim // heads)

        self.ln1 = nn.LayerNorm(out_dim)
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, x, y):
        inp = torch.concat([x, y], dim=1)  # (128, 1024+x, 256)
        b, l, _ = inp.shape
        Q = self.wq(inp).reshape(b, l, self.num_heads, self.out_dim // self.num_heads).transpose(1, 2)
        K = self.wk(inp).reshape(b, l, self.num_heads, self.out_dim // self.num_heads).transpose(1, 2)
        V = self.wv(inp).reshape(b, l, self.num_heads, self.out_dim // self.num_heads).transpose(1, 2)
        # attention
        dist = torch.matmul(Q, K.transpose(2, 3)) * self.dk  # (Q * K^T) / dk
        dist = torch.softmax(dist, dim=-1)
        attn = torch.matmul(dist, V)

        attn = attn.transpose(1, 2).reshape(b, l, self.out_dim)  # (128, 1024+x, 256)
        x = inp + attn

        x = torch.sum(x, dim=1)  # (128, 256)
        x = self.ln1(x)  # batch norm
        x = self.proj(x) + x  # feed-forward
        x = self.ln2(x)
        return x


class FFN(nn.Module):
    def __init__(self, dim=128, ratio=4, dropout=0.1):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * ratio, dim)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x


class FFN2(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super(FFN2, self).__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        res = x
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x + res


class TransformerBlock(nn.Module):
    def __init__(self, dim=128, num_heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MHA = MHA(dim, dim, num_heads)  # 128 -> 128
        self.ln1 = nn.LayerNorm(dim)
        self.FFN = FFN(dim)  # 128 -> 128
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.MHA(x) + x  # multi-head attention
        x = self.ln1(x)  # batch norm
        x = self.FFN(x) + x  # feed-forward
        x = self.ln2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=2048):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super().__init__()
        self.mamba = Mamba2(d_model=hidden_size, d_state=64, d_conv=4, expand=2)
        self.norm2 = LlamaRMSNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.mamba(x)
        x = h
        h = self.mlp(self.norm2(x))
        return x + h


class Fea_Encoder(nn.Module):
    def __init__(self,
                 input_size,  # 1024
                 dense_layers,  # 4
                 sparse_layers,  # 2
                 num_experts,  # 4
                 dropout,  # 0.2
                 hidden_size=128,  # 128
                 output_dim=256,  # 512
                 ):
        super(Fea_Encoder, self).__init__()
        # 线性映射层
        hidden_size = 256

        self.input_layer = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.pe = PositionalEncoding(hidden_size, 0.1)
        self.Mamba = nn.Sequential(
            MambaLayer(hidden_size, 0.1),
            MambaLayer(hidden_size, 0.1),
            MambaLayer(hidden_size, 0.1),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(1024, output_dim),  # 1024 -> 512
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (128, 1024, 2)
        x = self.input_layer(x)  # (128, 1024, 128)
        x = self.pe(x)
        x = self.Mamba(x)  # (128, 1024, 128)
        # 更换聚合方法
        x = x.permute(0, 2, 1)  # (128, 128, 1024)
        x = torch.sum(x, dim=1)  # (128, 1024)
        x = self.output_layer(x)  # (128, 1024)
        return x, None


# ---------------------------------
class PositionalWiseFeedForward(nn.Module):
    """
    Fully-connected network
    """

    def __init__(self, model_dim=256, ffn_dim=2048, dropout=0.5):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x

        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        output = x
        return output


class multimodal_attention(nn.Module):
    """
    dot-product attention mechanism
    """

    def __init__(self, attention_dropout=0.5):
        super(multimodal_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.matmul(q, k.transpose(-2, -1))
        if scale:
            attention = attention * scale

        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int = 256, emb_dim: int = 256, num_heads: int = 8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = int(emb_dim // num_heads)
        self.num_heads = num_heads
        self.linear_k = nn.Linear(1, int(self.dim_per_head * num_heads), bias=False)
        self.linear_v = nn.Linear(1, int(self.dim_per_head * num_heads), bias=False)
        self.linear_q = nn.Linear(1, int(self.dim_per_head * num_heads), bias=False)

        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = nn.Linear(emb_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query
        query = query.unsqueeze(-1)
        key = key.unsqueeze(-1)
        value = value.unsqueeze(-1)
        # print("query.shape:{}".format(query.shape))

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        # batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        # print('key.shape:{}'.format(key.shape))

        # split by heads
        # print(num_heads, self.model_dim, dim_per_head)
        self.model_dim = int(self.model_dim)
        key = key.view(-1, num_heads, self.model_dim, dim_per_head)
        value = value.view(-1, num_heads, self.model_dim, dim_per_head)
        query = query.view(-1, num_heads, self.model_dim, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        attention = self.dot_product_attention(query, key, value,
                                               scale, attn_mask)

        attention = attention.view(-1, self.model_dim, dim_per_head * num_heads)
        # print('attention_con_shape:{}'.format(attention.shape))

        # final linear projection
        output = self.linear_final(attention).squeeze(-1)
        # print('output.shape:{}'.format(output.shape))
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output


class multimodal_fusion_layer(nn.Module):
    """
    A layer of fusing features
    """

    def __init__(self, model_dim=256, emb_dim=256, num_heads=8, ffn_dim=2048, dropout=0.5):
        super(multimodal_fusion_layer, self).__init__()
        self.attention_1 = MultiHeadAttention(model_dim, emb_dim, num_heads, dropout)
        self.attention_2 = MultiHeadAttention(model_dim, emb_dim, num_heads, dropout)

        self.feed_forward_1 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.feed_forward_2 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

        self.fusion_linear = nn.Linear(model_dim * 2, model_dim)

    def forward(self, image_output, text_output, attn_mask=None):
        output_1 = self.attention_1(image_output, text_output, text_output,
                                    attn_mask)

        output_2 = self.attention_2(text_output, image_output, image_output,
                                    attn_mask)

        # print('attention out_shape:{}'.format(output.shape))
        output_1 = self.feed_forward_1(output_1)
        output_2 = self.feed_forward_2(output_2)

        output = torch.cat([output_1, output_2], dim=1)
        output = self.fusion_linear(output)

        return output
