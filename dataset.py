import random
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import numpy as np
import settings

def create_dataset(df, split=0.8, max_count=1e10):
    # remove year 2008 and 2020
    horse_cols = ['horse_age', 'horse_weight', 'horse_speed_rating', 'horse_past_race',
                  'horse_exp_month', 'horse_past_race_dist', 'horse_weight_change',
                  'horse_earnings_cum', 'horse_earnings_avg5','horse_speed_rating_avg5',
                  'horse_life_win%', 'horse_earnings_avg3']
    jockey_cols = ['jockey_past_race', 'jockey_life_win#', 'jockey_life_win%', 
                   'jockey_earnings_cum', 'jockey_earnings_avg5', 'jockey_earnings_avg3']
    cat_cols = ['horse_sex_index', 'horse_lob_index']

    cols = horse_cols + jockey_cols + cat_cols

    # using data 2009 ~ 2018 (2019 for test set)
    df = df[(df.race_year < 2019) & (df.race_year > 2008)]
    #df['odd_single'] = np.log(df['odd_single']) - 1
    df = df[['race_id'] + cols + ['standing', 'prob_single']]

    assert df.isna().sum().sum() == 0

    R = min(df.race_id.nunique(), max_count)
    H = df.groupby('race_id')['race_id'].count().max()

    x_race = torch.zeros(R, H, df.shape[1]-1)  # less race_id
    x_horse_count = torch.zeros(R)

    #TODO: remove racing too small?
    #TODO: mask and number of horses in a race

    for i, (race_id, g) in tqdm(enumerate(df.groupby('race_id'))):
        g.pop('race_id')
        h = g.values.shape[0] # num horses
        x_race[i, :h, :] = torch.tensor(g.values)
        x_horse_count[i] = torch.tensor(h)
    
    D = (x_race, x_horse_count)

    print('Splitting train dataset...')
    # Split the training data into train (80%) and validation (20%) for model selection.
    np.random.seed(0)
    p = np.random.permutation(D[0].shape[0])

    split_ix = int(len(p) * 0.8)
    idx_train = torch.tensor(p[:split_ix])
    idx_val = torch.tensor(p[split_ix:])

    for d in D:
        assert d.isnan().sum() == 0

    D_train = tuple([d[idx_train] for d in D])
    D_val = tuple([d[idx_val] for d in D])

    torch.save(D_train, settings.train_data_path)
    torch.save(D_val, settings.val_data_path)

    return D_train, D_val

def get_datasets():
    train_ds = torch.load(settings.train_data_path)
    val_ds = torch.load(settings.val_data_path)

    return TensorDataset(*train_ds), TensorDataset(*val_ds)

if __name__ == '__main__':
    o = torch.load(settings.prep_data_path)
    create_dataset(o['data'])
