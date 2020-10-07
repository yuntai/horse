import torch
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

train_dataset, val_dataset = dataset.get_datasets()

max_num_horses = train_dataset.tensors[1].max().long()
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=True)
m = torch.load("model.ckpt")
tot_cnt = 0
hit_cnts = [0]*5
hit_cnts_y = [0]*5
hit_cnts_z = [0]*5
torch.set_grad_enabled(False)

for x, cnt in tqdm(val_loader):
    x = x.to('cuda')
    cnt = cnt.to('cuda')
    bsz = x.size(0)
    rix = torch.stack([torch.randperm(max_num_horses) for _ in range(bsz)]).to('cuda')
    x = x[torch.arange(bsz)[:,None,None], rix[...,None], torch.arange(x.size(-1))]
    mask = (rix < cnt[:,None]).float()
    mask = mask.to('cuda')
    feats, toks, standing, y = x.split([18, 2, 1, 1], dim=-1)
    y.squeeze_()
    standing = standing.long().squeeze()
    standing *= mask.long()
    standing[standing==0] = 10000
    toks = toks.type(torch.long)
    pred = m(feats, toks, mask)

    top5_pred = torch.argsort(pred*mask, descending=True)[:,:5].cpu().numpy()
    top2_y = torch.argsort(y*mask, descending=True)[:,:2].cpu().numpy()
    top5_y = torch.argsort(y*mask, descending=True)[:,:5].cpu().numpy()
    top2_gt = torch.nonzero(standing<=2, as_tuple=True)[1]

    if top2_gt.size(0) != bsz * 2: # tie
        continue

    top2_gt = top2_gt.reshape(bsz,-1).cpu().numpy()
    for p, g, q, r in zip(top5_pred, top2_gt, top2_y, top5_y):
        inte_ = set(p) & set(g)
        hit_cnts[len(inte_)] += 1
        inte_ = set(p) & set(q)
        hit_cnts_y[len(inte_)] += 1
        inte_ = set(g) & set(r)
        hit_cnts_z[len(inte_)] += 1
    tot_cnt += bsz

print(tot_cnt)

print(hit_cnts)
print(np.array(hit_cnts)/tot_cnt*100)

print(hit_cnts_y)
print(np.array(hit_cnts_y)/tot_cnt*100)

print(hit_cnts_z)
print(np.array(hit_cnts_z)/tot_cnt*100)
