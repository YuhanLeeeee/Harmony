import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import os
import pickle
import pandas as pd


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train(model, dataloader, optimizer, scheduler, criterion, device, mean, std):
    model.train()
    running_loss = 0.0

    mp = {}
    mp["r"] = []
    mp["p"] = []
    mp["grd"] = []
    mp["msk"] = []

    for batch in tqdm(dataloader, desc='training...', total=len(dataloader)):
        # SMILES graph human-defined label
        smiles_batch, graph_batch, fp_batch, labels = batch  # token_smiles, graph, fingerprint, yield
        labels = (labels - mean) / std

        optimizer.zero_grad()
        labels = labels.to(device)
        smiles_batch = smiles_batch.to(device)
        graph_batch = [(rmols.to(device), pmols.to(device)) for rmols, pmols in graph_batch]
        fp_batch = fp_batch.to(device)
        fp_batch.requires_grad_(True)

        pred, logvar, cl_loss, _, outs = model(smiles_batch, graph_batch, fp_batch)
        pred = torch.where(torch.isnan(pred), torch.full_like(pred, 0), pred)
        pred = torch.where(torch.isinf(pred), torch.full_like(pred, 0), pred)
        pred = pred.squeeze(-1)

        # L_prefer
        num_modal = 3  # modality types
        margin_loss = 0.0
        for modality in range(num_modal):
            out_cur = outs[modality]
            out_cur = out_cur.squeeze(-1)
            out_cur = torch.where(torch.isnan(out_cur), torch.full_like(out_cur, 0), out_cur)
            margin_loss += criterion(labels, out_cur)
        loss_margin = margin_loss * 2.0

        loss = criterion(pred, labels)
        # 0.05、0.08、0.1、0.15、0.2、0.3、0.5
        # 0.05、0.1、0.15、0.2、0.25、0.3、0.5
        loss = 1.0 * loss.mean() \
               + 0.1 * (loss * torch.exp(-logvar) + logvar).mean() \
               + 0.1 * cl_loss \
               + 0.2 * loss_margin.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.detach().item()

    return running_loss / len(dataloader)


def valid(model, dataloader, criterion, device, mean, std):
    model.eval()
    y_true = []
    y_pred = []
    total_phi_grp = 0.0
    total_phi_fp = 0.0
    total_phi_seq = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='validation...'):
            smiles_batch, graph_batch, fp_batch, labels = batch

            y_true.append(labels.cpu().numpy())
            smiles_batch = smiles_batch.to(device)
            graph_batch = [(rmols.to(device), pmols.to(device)) for rmols, pmols in graph_batch]
            fp_batch = fp_batch.to(device)
            pred, _, _, outs, _ = model(smiles_batch, graph_batch, fp_batch)
            pred = torch.where(torch.isnan(pred), torch.full_like(pred, 0), pred)
            assert not torch.any(torch.isnan(pred)), "Model output contains NaN values!"

            pred = pred.cpu().numpy() * std.numpy() + mean.numpy()

            y_pred.append(pred)

            # contribution of each modality
            pred = torch.from_numpy(pred)
            outs[0] = outs[0].cpu().numpy() * std.numpy() + mean.numpy()  # wo graph
            outs[1] = outs[1].cpu().numpy() * std.numpy() + mean.numpy()  # wo human
            outs[2] = outs[2].cpu().numpy() * std.numpy() + mean.numpy()  # wo smiles
            outs[0] = torch.from_numpy(outs[0])
            outs[1] = torch.from_numpy(outs[1])
            outs[2] = torch.from_numpy(outs[2])

            phi_grp = count_v(pred, labels, 3) - count_v(outs[0], labels, 2)
            phi_fp = count_v(pred, labels, 3) - count_v(outs[1], labels, 2)
            phi_seq = count_v(pred, labels, 3) - count_v(outs[2], labels, 2)

            total_phi_grp += phi_grp.mean()
            total_phi_fp += phi_fp.mean()
            total_phi_seq += phi_seq.mean()

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    print(mae, mse, r2)

    total_phi = np.exp(total_phi_fp) + np.exp(total_phi_seq) + np.exp(total_phi_grp)

    return r2, [total_phi_grp, total_phi_fp, total_phi_seq], [np.exp(total_phi_grp) / total_phi,
                                                              np.exp(total_phi_fp) / total_phi,
                                                              np.exp(total_phi_seq) / total_phi]


def test(model, dataloader, device, mean, std):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='testing...'):
            smiles_batch, graph_batch, fp_batch, labels = batch

            y_true.append(labels.cpu().numpy())
            smiles_batch = smiles_batch.to(device)
            graph_batch = [(rmols.to(device), pmols.to(device)) for rmols, pmols in graph_batch]
            fp_batch = fp_batch.to(device)
            pred, _, _, _, _ = model(smiles_batch, graph_batch, fp_batch)
            pred = pred.cpu().numpy() * std.numpy() + mean.numpy()

            y_pred.append(pred)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f'MAE: {mae:.3f}, MSE: {mse:.3f}, R2: {r2:.3f}')
    return r2, y_true, y_pred


def do_CL(X, Y, logit_scale):  # L_info
    criterion = torch.nn.CrossEntropyLoss()
    B = X.size(0)
    logits = torch.matmul(X, Y.T) * logit_scale  # [128, 256] * [256, 128] -> [128, 128]
    labels = torch.arange(B, device=logits.device)  # [0, 1, 2, ..., 127]
    CL_loss = criterion(logits, labels).mean()
    return CL_loss


def count_v(yc, y, C, threshold=0.1):
    delta = torch.abs(yc - y)
    mask = delta > threshold
    delta = (delta + 1e-5) / threshold  # [0,1]
    delta = delta.masked_fill(mask, 1)
    delta = -torch.log(delta)
    delta = torch.where(torch.isnan(delta), torch.full_like(delta, 1), delta)
    delta = torch.where(torch.isinf(delta), torch.full_like(delta, 1), delta)
    mask = delta > 1
    res = delta.masked_fill(mask, 1)
    return res * C
