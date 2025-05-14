import random

import torch
import torch.nn as nn
from .encoders import *
from einops import repeat, rearrange
import numpy as np
from utils import do_CL

from efficient_kan import KAN
import copy


class CLME(nn.Module):
    def __init__(self,
                 node_in_feats, edge_in_feats,
                 g_hidden_size, num_step_mp, num_step_set2set, num_layer_set2set, g_output_dim,
                 model_path, vocab_size,
                 s_embed_dim, num_heads, num_layers, context_length, s_output_dim, cl_hidden_dim=256
                 ):
        super(CLME, self).__init__()

        self.mpnn = MolEncoder(node_in_feats, edge_in_feats,  # 2d graph encoder
                               hidden_size=g_hidden_size, num_step_mp=num_step_mp,
                               num_step_set2set=num_step_set2set, num_layer_set2set=num_layer_set2set,
                               output_dim=g_output_dim)
        self.transformer = SMILES_Encoder(  # smiles encoder
            model_path=model_path,
            vocab_size=vocab_size, embed_dim=s_embed_dim, num_heads=num_heads,
            num_layers=num_layers, context_length=context_length,
            output_dim=s_output_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.final_proj = nn.Sequential(
            nn.Linear(g_output_dim * 2, g_output_dim),
            nn.BatchNorm1d(g_output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def process_graph(self, graph_ls, seq_feats):  # molecular-level fusion
        reactant_features = []
        product_features = []

        for idx, (reactant_batch, product_batch) in enumerate(graph_ls):
            if seq_feats is not None:
                encoded_reactants = self.mpnn(reactant_batch, seq_feats[idx])  # (9, 1024)
                encoded_products = self.mpnn(product_batch, seq_feats[idx])  # (1, 1024)
            else:
                encoded_reactants = self.mpnn(reactant_batch, seq_feats)  # (9, 1024)
                encoded_products = self.mpnn(product_batch, seq_feats)  # (1, 1024)
            reactant_feature = encoded_reactants.sum(dim=0)  # (1024)
            product_feature = encoded_products.sum(dim=0)  # (1024)
            reactant_features.append(reactant_feature)
            product_features.append(product_feature)

        reactant_features = torch.stack(reactant_features)  # (128, 1024)
        product_features = torch.stack(product_features)
        # mlp
        graph_feats = torch.cat([reactant_features, product_features], 1)  # (128, 2048)
        graph_feats = self.final_proj(graph_feats)  # 2048 -> 1024
        return graph_feats  # (128, 1024)


class Harmony(nn.Module):
    def __init__(self,
                 mlp_input_size, node_in_feats, edge_in_feats,  # 1024， 9， 3
                 mlp_hidden_size, dense_l, spar_l, num_exps, mlp_drop, mlp_out_size,  # moe
                 g_hidden_size, num_step_mp, num_step_set2set, num_layer_set2set, g_output_dim,
                 vocab_size, s_embed_dim, num_heads, num_layers, context_length, s_output_dim,
                 predict_hidden_dim, prob_dropout,
                 model_path='./ChemBERTa',
                 ):
        super(Harmony, self).__init__()

        # fingerprints encoder
        self.trans = Fea_Encoder(input_size=mlp_input_size,  # 1024
                                 hidden_size=mlp_hidden_size, dense_layers=dense_l, sparse_layers=spar_l,
                                 num_experts=num_exps, dropout=mlp_drop, output_dim=mlp_out_size)
        # SMILES and 2D graph encoder
        self.clme = CLME(node_in_feats,  # 9
                         edge_in_feats,  # 3
                         model_path=model_path,
                         g_hidden_size=g_hidden_size, num_step_mp=num_step_mp,
                         num_step_set2set=num_step_set2set, num_layer_set2set=num_layer_set2set,
                         g_output_dim=g_output_dim, vocab_size=vocab_size, s_embed_dim=s_embed_dim, num_heads=num_heads,
                         num_layers=num_layers, context_length=context_length, s_output_dim=s_output_dim)

        self.final_proj = nn.Sequential(
            # MLP
            nn.Linear(
                g_output_dim + mlp_out_size + s_output_dim,
                predict_hidden_dim),  # 3072 -> 512
            nn.BatchNorm1d(predict_hidden_dim),
            nn.PReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_dim, predict_hidden_dim),
            nn.BatchNorm1d(predict_hidden_dim),
            nn.PReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_dim, 64),  # 512->64
            # KAN
            KAN([64, 2], grid_size=12)  # 12,591,530
        )

    def forward(self, smiles, mols, input_feats):
        # (128, 512)
        human_feats, fp_out = self.trans(input_feats)  # fingerprints feature
        # (128, 1024)
        seq_feats = self.clme.transformer(smiles)  # SMILES feature
        # (128, 1024)
        graph_feats = self.clme.process_graph(mols, seq_feats)  # 2D graph feature

        concat_feats = torch.cat([graph_feats, human_feats, seq_feats], dim=1)  # concat
        out = self.final_proj(concat_feats)  # late fuse -> (128, 64)

        # yield results
        mean = out[:, 0]
        logvar = out[:, 1]

        # ----------- L_info -----------
        graph_features = graph_feats / graph_feats.norm(dim=1, keepdim=True)
        human_feats = human_feats / human_feats.norm(dim=1, keepdim=True)
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        cl_loss = do_CL(graph_features, human_feats, logit_scale) + do_CL(human_feats, graph_features,
                                                                          logit_scale)
        # ----------- L_info -----------

        # ----------- Predict yield using single modality(for computing L_prefer). -----------
        out_grp = self.final_proj(
            torch.cat([graph_feats, torch.zeros_like(human_feats), torch.zeros_like(seq_feats)], dim=1))
        out_hum = self.final_proj(
            torch.cat([torch.zeros_like(graph_feats), human_feats, torch.zeros_like(seq_feats)], dim=1))
        out_seq = self.final_proj(
            torch.cat([torch.zeros_like(graph_feats), torch.zeros_like(human_feats), seq_feats], dim=1))
        outs = [out_grp[:, 0], out_hum[:, 0], out_seq[:, 0]]
        # outs = None
        # ----------- Predict yield using single modality(for computing L_prefer). -----------

        # ----------- Counterfactual Reasoning-based Evaluation Method(for computing modality contribution). -----------
        self.final_proj_copy = copy.deepcopy(self.final_proj)
        wo_grp = self.final_proj_copy(torch.cat([torch.zeros_like(graph_feats), human_feats, seq_feats], dim=-1))
        wo_hum = self.final_proj_copy(torch.cat([graph_feats, torch.zeros_like(human_feats), seq_feats], dim=-1))
        wo_seq = self.final_proj_copy(torch.cat([graph_feats, human_feats, torch.zeros_like(seq_feats)], dim=-1))
        wo_outs = [wo_grp[:, 0], wo_hum[:, 0], wo_seq[:, 0]]
        # ----------- Counterfactual Reasoning-based Evaluation Method(for computing modality contribution). -----------

        return mean, logvar, cl_loss, wo_outs, outs
