import torch
import settings
from torch.utils.data import TensorDataset, DataLoader
import argparse
from dataset import get_datasets
from graph_transformer import GraphTransformer
import numpy as np
import itertools
import torch.nn as nn
from torch import optim
import tqdm

parser = argparse.ArgumentParser(description='Graph Transformer on Horse Racing outcome')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_epoch', type=int, default=1000, help='upper epoch limit')

parser.add_argument('--d_model', type=int, default=640, help='model dimension')
parser.add_argument('--n_layer', type=int, default=14, help='number of total layers')
parser.add_argument('--n_head', type=int, default=10, help='number of heads')
parser.add_argument('--d_inner', type=int, default=3800, help='inner dimension in posFF')
parser.add_argument('--d_embed', type=int, default=128, help='inner dimension in posFF')
parser.add_argument('--final_dim', type=int, default=280, help='final layer hidden dimension')
parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate (0.0001|5 for adam|sgd)')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--decay_rate', type=float, default=0.5, help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--patience', type=int, default=5, help='patience')
parser.add_argument('--lr_min', type=float, default=0.0, help='minimum learning rate during annealing')

parser.add_argument('--dropout', type=float, default=0.03, help='global dropout rate (applies to residual blocks in transformer)')
parser.add_argument('--dropatt', type=float, default=0.0, help='attention probability dropout rate')
parser.add_argument('--final_dropout', type=float, default=0.04, help='final layer dropout rate')
parser.add_argument('--wnorm', action='store_true', help='use weight normalization')

parser.add_argument('--cuda', type=bool, default=True, help='use CUDA')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--eta_min', type=float, default=1e-7,
                    help='min learning rate for cosine scheduler')

args = parser.parse_args()

train_dataset, val_dataset = get_datasets()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

model = GraphTransformer(dim=args.d_model, n_layers=args.n_layer, d_inner=args.d_inner, n_toks=[3, 12],
                         d_embed=args.d_embed, final_dim=args.final_dim, dropout=args.dropout, n_feat=18,
                         dropatt=args.dropatt, final_dropout=args.final_dropout, n_head=args.n_head,
                         wnorm=args.wnorm).to(device)

bce_loss = nn.BCEWithLogitsLoss(reduction='none')
loss_func = nn.MSELoss(reduction='none')
def criterion(pred, y, mask, x):
    loss = loss_func(pred, y)
    res = (loss * mask).sum()/((mask>0).sum())
    return res

args.max_step = args.max_epoch * len(train_loader)

# initlize optimizer and lr scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay_rate,
                                                     patience=args.patience, min_lr=args.lr_min)
elif args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch, eta_min=args.eta_min)

#TODO: prper union with val_dataset
max_num_horses = train_dataset.tensors[1].max().long()

def accuracy(pred, y, mask, threshold=0.7):
    pred = (torch.sigmoid(pred) > threshold).long()
    return (pred * y).sum() / 5. / pred.size(0)

ix = 0
start_epoch = 0
train_step = 0
min_val_loss = float('inf')
for epoch_i in range(start_epoch, args.max_epoch):
    losses = []
    accs = []
    model.train()
    with torch.enable_grad():
        for _batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            x, cnt = (b.to(device) for b in _batch)

            bsz = x.size(0)
            rix = torch.stack([torch.randperm(max_num_horses) for _ in range(bsz)])
            # with gather possible?
            x = x[torch.arange(bsz)[:,None,None], rix[...,None], torch.arange(x.size(-1))]

            mask = (rix < cnt[:,None]).float()

            feats, toks, y = x.split([18, 2, 1], dim=-1)
            toks = toks.type(torch.long)
            pred = model(feats, toks, mask)
            loss = criterion(pred, y.squeeze(), mask, x)
            #acc = accuracy(pred, y.squeeze(), mask)
            #accs.append(acc.detach().cpu())
            loss.backward()
            losses.append(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_step += 1

    model.eval()
    with torch.no_grad():
        val_losses = []
        for _batch in tqdm.tqdm(val_loader):
            x, cnt = (b.to(device) for b in _batch)

            bsz = x.size(0)
            rix = torch.stack([torch.randperm(max_num_horses) for _ in range(bsz)])
            # with gather possible?
            x = x[torch.arange(bsz)[:,None,None], rix[...,None], torch.arange(x.size(-1))]

            mask = (rix < cnt[:,None]).float()

            feats, toks, y = x.split([18, 2, 1], dim=-1)
            toks = toks.type(torch.long)
            pred = model(feats, toks, mask)
            loss = criterion(pred, y.squeeze(), mask, x)
            val_losses.append(loss.detach().cpu())

    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    tr_loss = np.array(losses).mean()
    val_loss = np.array(val_losses).mean()
    if val_loss < min_val_loss:
        torch.save(model, 'model.ckpt')
        print(f"Saving model ...")
        min_val_loss = val_loss
    print(f"epoch {epoch_i} loss({tr_loss:.7f}) lr({lr:.7f}) val_loss({val_loss:.7f})")
