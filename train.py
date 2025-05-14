import os

import torch
from torch.utils.data import DataLoader, random_split
from model.mmmodel import *
from dataset import ReactionDataset, collate_fn, tokenizer
import pickle
from utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import shutil
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "true"

config = load_config('./config.yaml')
train_config = config['training']

# dataset
if 'SM' in train_config['dataset_path']:
    if 'new' in train_config['dataset_path']:
        used_dataset = 'SM_new'
    else:
        used_dataset = 'SM'
elif 'BH' in train_config['dataset_path']:
    used_dataset = 'BH'
elif 'USPTO' in train_config['dataset_path'] or 'ORD' in train_config['dataset_path']:
    used_dataset = 'USPTO'
elif '622' in train_config['dataset_path']:
    used_dataset = 'ACR622'
else:
    used_dataset = 'ACR'

# model config
mlp_config = config['mlp_model']
gnn_config = config['graph_model']
seq_config = config['smiles_model']
predictor_config = config['predictor']
patience = train_config['patience']
exp_name = train_config['exp_name']

if torch.cuda.is_available():
    device = torch.device(f"cuda:{train_config['cuda']}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.get_device_capability(device)[0] >= 8:
        print('GPU supports bfloat16.')
    else:
        print('GPU does not support bfloat16.')
else:
    print('No GPU is available!')

with open(train_config['dataset_path'], 'rb') as f:
    data = pickle.load(f)  # load data

vocab_length = tokenizer.vocab_size

# model save path
model_save_dir = f'./checkpoint/{exp_name}'
src_file = ["./config.yaml",
            "./train.py",
            "./utils.py",
            "./model/encoders.py",
            "./model/mmmodel.py"]
model_path = 'ChemBERTa'

model = Harmony(1024,
                9,  # node_in_feats(gnn)
                3,  # edge_in_feats(gnn)
                # moe
                mlp_hidden_size=mlp_config['mlp_hidden_size'], dense_l=mlp_config['dense_layers'],
                spar_l=mlp_config['sparse_layers'], num_exps=mlp_config['num_experts'],
                mlp_drop=mlp_config['dropout_ratio'], mlp_out_size=mlp_config['output_dim'],
                # gnn
                g_hidden_size=gnn_config['hidden_size'], num_step_mp=gnn_config['num_step_mp'],
                num_step_set2set=gnn_config['num_step_set2set'], num_layer_set2set=gnn_config['num_layer_set2set'],
                g_output_dim=gnn_config['output_dim'],
                # seq
                vocab_size=vocab_length,
                s_embed_dim=seq_config['embed_dim'], num_heads=seq_config['num_heads'],
                num_layers=seq_config['num_layers'], context_length=seq_config['context_length'],
                s_output_dim=seq_config['output_dim'],
                # mlp--predictor
                predict_hidden_dim=predictor_config['hidden_size'],
                prob_dropout=predictor_config['dropout_ratio'],
                model_path=model_path
                ).to(device)  # 模型

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')  # mlp:20,673,100, kan:37,194,312
total_trainable_params = sum(p.numel() for p in model.parameters())
print(f'total {total_trainable_params:,} parameters.')

criterion = nn.MSELoss(reduction='none')
optimizer = optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
lr_scheduler = CosineAnnealingLR(optimizer, T_max=train_config['epochs'], eta_min=5e-5)

train_ratio = train_config['train_ratio']
validation_exist = True
if 'val' not in data.keys():
    validation_exist = False
if 'split' in train_config['dataset_path'] or 'USPTO' in train_config['dataset_path']:
    print('split before!')
    train_data = ReactionDataset(data['train'], 'train', train_ratio)
    val_data = None
    if validation_exist:
        val_data = ReactionDataset(data['val'], 'val')
    test_data = ReactionDataset(data['test'], 'test')
    train_mean, train_std = train_data.calculate_yield_stats()
else:
    print('split now!')
    print(len(data['rxn']))
    reaction_dataset = ReactionDataset(data)
    dataset_size = len(reaction_dataset)
    validation_exist = False
    train_size = int(0.7 * dataset_size)
    test_size = dataset_size - train_size

    train_data, test_data = random_split(reaction_dataset, [train_size, test_size])
    val_data = None
    train_mean, train_std = train_data.dataset.calculate_yield_stats(train_data.indices)

print(train_mean, train_std)
print(f"Training set size: {len(train_data)}")
if validation_exist:
    print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

trainloader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)
valloader = None
if validation_exist:
    valloader = DataLoader(val_data, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn)
testloader = DataLoader(test_data, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn)

best_model = 0.1
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

for file in src_file:
    shutil.copy(file, model_save_dir)
    print(f"file {file} have saved!")

with torch.cuda.device(device):
    for epoch in range(train_config['epochs']):
        trainloss = train(model, trainloader, optimizer, lr_scheduler, criterion, device, train_mean,
                          train_std)  # train
        if validation_exist:
            validr2, phis, norm_phis = valid(model, valloader, criterion, device, train_mean,
                                             train_std)  # validation (for ACR)
        else:
            validr2, phis, norm_phis = valid(model, testloader, criterion, device, train_mean,
                                             train_std)  # The validation set doesn't exist, so directly conduct the test (for BH and SM datasets)

        if validr2 > best_model:
            early_count = 0
            best_model = validr2
            torch.save(model.state_dict(), f'{model_save_dir}/{used_dataset}_model.pth')
        print(
            "Epoch: {} Train Loss: {:.3f} Valid R2: {:.3f} early_count: {}, phi_grp: {:5f}, phi_fp: {:5f}, phi_seq: {:5f}".format(
                epoch + 1, trainloss, validr2, early_count, phis[0], phis[1], phis[2]))

    model.load_state_dict(torch.load(f'{model_save_dir}/{used_dataset}_model.pth', map_location=device),
                          strict=False)  # load model

    validr2, phis, norm_phis = valid(model, testloader, criterion, device, train_mean,
                                     train_std)  # testloader
