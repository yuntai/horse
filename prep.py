from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

TRACK_TAKE = 0.3

import settings
from name_mapping import NAME_MAPPER

def _create_embedding(series):
    types = sorted(series.unique().tolist())
    assert "<None>" not in types
    emb_index = dict(zip(["<None>"] + types , range(len(types)+1)))

    return emb_index

def create_embedding_(df, cols):
    embeddings = {}
    for c in cols:
        emb_index = _create_embedding(df[c])
        df[c + '_index'] = df[c].apply(lambda x: emb_index[x])
        embeddings[c] = emb_index
    return embeddings

def check_data(df):
    __standing90 = df.standing < 90
    # standing code
    #   91: 실격
    #   92: 주행중지
    #   95: 출전취소
    #   93: 출전제외
    #   94: 출전제외
    # we can use odd for standing 91/92, otherwise odd's are null
    assert df[((df.standing==91) | (df.standing==92)) & (df.odd_single.isna())].shape[0] == 0
    assert df[((df.standing==91) | (df.standing==92)) & (df.odd_double.isna())].shape[0] == 0
    assert df[(df.standing >= 93) & ~df.odd_single.isna()].shape[0] == 0
    assert df[(df.standing >= 93) & ~df.odd_double.isna()].shape[0] == 0
    assert df[((df.standing==91) | (df.standing==92)) & (df.odd_double.isna())].shape[0] == 0
    assert df.rec_final.isna().sum() == 0

    # https://www.chosun.com/site/data/html_dir/1999/06/09/1999060970476.html
    # S1F ( record 200m from start)
    # G1F ( record 200m from finish)
    # G3F (        600m from finish)
    # 1000m: 3C ~= s1f
    # 1300m:
    # 1400m: s1f ~= 2c
    # 1700m: s1f ~= 1c
    # 1800m
    # 1900m
    # 2000m
    # 2300m
    __dist1400 = df.race_dist <= 1400
    assert df[__dist1400 & (df.rec_1c.notna()|df.rec_2c.notna())].shape[0] == 0
    assert df[__dist1400 & __standing90 & (df.rec_3c.isna()|df.rec_4c.isna())].shape[0] == 0

    assert df.loc[(df['rec_s1f'].isna()) & (df.standing < 90)].shape[0] == 0
    assert df.loc[(df['rec_1fg'].isna()) & (df.standing < 90)].shape[0] == 0
    assert df.loc[(df['rec_3fg'].isna()) & (df.standing < 90)].shape[0] == 0

def load_data():
    """ load data/rename columns & values """
    df = pd.read_csv(settings.rawdata/'seoulracing.csv')
    df = df.rename(NAME_MAPPER, axis=1)
    track_condition_mapper = {'양':'A','다':'B','건':'C','불':'D','포':'E'}
    df.track_condition = df.track_condition.map(track_condition_mapper)
    def race_level_mapper(l):
        # ['국5', '국6', '혼5', '국2', '국4', '국3', '혼3', '혼4', '혼1', '혼2', '국1', '국오', '혼오']
        __m = {'국': 'K', '혼': 'M'}
        return __m[l[0]] + ('O' if l[1]=='오' else l[1])
    df.race_level = df.race_level.map(race_level_mapper)
    weathers = ['맑', '흐', '눈', '비', '안', '강']
    weather_mapper = dict(zip(weathers, 'ABCDEFG'))
    df.weather = df.weather.map(weather_mapper)
    df.race_dow = df.race_dow.map(dict(zip('토일금', ['SAT', 'SUN', 'FRI'])))

    cols_to_drop = ['horse_name', 'trainer_name', 'jockey_name', 'owner_name', 'race_time']
    df.drop(cols_to_drop, axis=1, inplace=True)

    # set unique race_id
    race_id = df['race_date'].apply(str) + '_' + df['race_number'].apply(lambda x: "%02d"%x)
    df.insert(0, 'race_id', race_id)
    df = df.sort_values(by='race_id', axis=0).reset_index(drop=True)

    # make year and month columns
    df['race_date'] = df['race_date'].astype('str')
    df['race_year'] = df['race_date'].apply(lambda x:x[:4]).astype('int')
    df['race_month'] = df['race_date'].apply(lambda x:x[4:6]).astype('int')
    df['month'] = (df['race_year'] - 2008) * 12 + df['race_month']

    # remove races that not having run
    race_ids_no_race = []
    race_ids_no_race_with_odd = []
    for race_id, gdf in df.groupby('race_id'):
        if gdf.standing.isna().all():
            race_ids_no_race.append(race_id)
            if gdf.odd_single.notna().all() and gdf.odd_double.notna().all():
                race_ids_no_race_with_odd.append(race_id)
    # number of no races with odds avialble = 2 (let's ignore)
    df = df.set_index('race_id').drop(race_ids_no_race).reset_index(drop=False)

    # fix standing
    df.loc[df['standing'].isna(), 'standing'] = 99
    df.loc[(df['rec_s1f'].isna()|df['rec_1fg'].isna()|df['rec_3fg'].isna()) & (df.standing < 90), 'standing'] = 99
    df['standing'] = df.standing.astype('int')

    # remove disqualied horses
    df = df[df.standing < 90]

    # number of races after
    num_races = len(df.groupby(df.race_id))
    num_horses = len(df.groupby(df.horse_id))

    # convert record format to float
    def __convert_rec(s):
        s = s.str.split(':', expand=True).astype('float')
        return s[0]*60 + s[1]
    rec_cols = [c for c in df.columns if c.startswith('rec_')]
    for col in rec_cols:
        __notna = df[col].notna()
        df.loc[__notna, col] = __convert_rec(df.loc[__notna, col])

    track_records = pd.read_csv(settings.rawdata/'track_records.csv')
    track_records = track_records.set_index('dist').apply(__convert_rec).reset_index()

    # just one horse weight missing
    assert df[df.horse_weight.isna()].shape[0] == 1
    __filter = df.horse_id == df[df.horse_weight.isna()].horse_id.iloc[0]
    s = df.loc[__filter, 'horse_weight']
    t = (s.shift(1)+s.shift(-1))/2
    df.loc[__filter, 'horse_weight'] = s.fillna(t)

    check_data(df)

    return df, track_records

def prep():
    df, track_records = load_data()

    print("total num horses=", df.horse_id.nunique())
    print("total num races=", df.race_id.nunique())

    # TODO: any anomaly with standing<90?
    #__notna = df['rec_s1f'].notna()
    #df.loc[__notna, 'record_s1f'] = df.loc[__notna, 'record_s1f'].apply(__convert_rec).clip(0, 20.)

    # feature: speed rating
    assert df.rec_final.isna().sum() == 0
    print("processing horse speed rating...")
    for row in track_records.itertuples():
        # ala `SEARCHING FOR POSITIVE RETURNS AT THE TRACK` 
        __mask = df.race_dist == row.dist
        df.loc[__mask, 'horse_speed_rating'] = 100. - (df.loc[__mask, 'rec_final'] - row.best)/0.2
        # TODO: need to adjust for differnt tracks
        # TODO: spped rating for sections?

    # past race (total & distance)
    # TODO: past disqualification - don't care for now
    print("processing horse features...")
    for horse_id, g in tqdm(df.groupby('horse_id')):
        df.loc[g.index, 'horse_past_race'] = np.arange(g.shape[0])
        df.loc[g.index, 'horse_exp_month'] = df.loc[g.index, 'month'] - g.month.min()
        past_race_dist = pd.crosstab(g.race_id, g.race_dist).cumsum().shift(1).fillna(0).astype('int')
        past_race_dist = past_race_dist.unstack().reset_index(name='past_race_dist')
        df.loc[g.index, 'horse_past_race_dist'] = g.reset_index().merge(past_race_dist, on=['race_id','race_dist'],
                                                                      how='left').set_index('index')['past_race_dist'] # keeping index
        weight_changes = g['horse_weight'].diff()
        df.loc[g.index, 'horse_weight_change'] = weight_changes.fillna(weight_changes.mean())
        #TODO: indicator variable first run?
        #TODO: chaceck standing <= 5
        df.loc[g.index, 'horse_life_win%'] = ((g.standing==1).cumsum().shift(1) / df.loc[g.index].horse_past_race).fillna(0)
        #df.loc[g.index, 'life_win'] = (g.standing==1).cumsum().shift(1).fillna(0).astype('int')

        # earnings
        earnings = g.apply(lambda x: x[f'prize_{x.standing}'] if x.standing <= 5 else 0, axis=1)
        df.loc[g.index, 'horse_earnings_cum'] = earnings.cumsum().shift(1).fillna(0)
        df.loc[g.index, 'horse_earnings_avg3'] = earnings.rolling(3).mean().shift(1)
        df.loc[g.index, 'horse_earnings_avg5'] = earnings.rolling(5).mean().shift(1)
        cnt = min(5, g.shape[0])
        backfill = (earnings.cumsum().shift(1)[:cnt]/np.arange(cnt)).fillna(method='bfill')
        df.loc[g.index[:5], 'horse_earnings_avg5'] = backfill.values
        df.loc[g.index[:3], 'horse_earnings_avg3'] = backfill[:3].values

        df.loc[g.index, 'horse_speed_rating_avg5'] = g.rolling(5)['horse_speed_rating'].mean().shift(1)
        cnt = min(5, g.shape[0])
        backfill = (g['horse_speed_rating'].shift(1)[:cnt]/np.arange(cnt)).fillna(method='bfill')
        df.loc[g.index[:5], 'horse_speed_rating_avg5'] = backfill.values

    # horse with only ran once
    df.horse_weight_change.fillna(0, inplace=True)
    df.horse_earnings_avg3.fillna(0, inplace=True)
    df.horse_earnings_avg5.fillna(0, inplace=True)
    df['horse_speed_rating_avg5'].fillna(df['horse_speed_rating'], inplace=True)

    print("processing jockey features...")
    for jockey_id, g in tqdm(df.groupby('jockey_id')):
        df.loc[g.index, 'jockey_past_race'] = np.arange(g.shape[0])
        df.loc[g.index, 'jockey_life_win#'] = (g.standing==1).cumsum().shift(1).fillna(0)
        df.loc[g.index, 'jockey_life_win%'] = (df.loc[g.index, 'jockey_life_win#']/df.loc[g.index, 'jockey_past_race']).fillna(0)
        earnings = g.apply(lambda x: x[f'prize_{x.standing}'] if x.standing <= 5 else 0, axis=1)
        df.loc[g.index, 'jockey_earnings_cum'] = earnings.cumsum().shift(1).fillna(0)
        df.loc[g.index, 'jockey_earnings_avg5'] = earnings.rolling(5).mean().shift(1)
        df.loc[g.index, 'jockey_earnings_avg3'] = earnings.rolling(3).mean().shift(1)
        cnt = min(5, g.shape[0])
        backfill = (earnings.cumsum().shift(1)[:cnt]/np.arange(cnt)).fillna(method='bfill')
        df.loc[g.index[:5], 'jockey_earnings_avg5'] = backfill.values
        df.loc[g.index[:3], 'jockey_earnings_avg3'] = backfill[:3].values
        #TODO: jockey_past_dist_race?
        #TODO: jockey spped rating?
    
    # jockey run only once
    df.jockey_earnings_avg3.fillna(0, inplace=True)
    df.jockey_earnings_avg5.fillna(0, inplace=True)

    print("processing race features...")
    for race_id, g in df.groupby('race_id'):
        #df.loc[g.index, 'prob_single'] = 1. / (1. + g['odd_single'])
        df.loc[g.index, 'prob_single'] = 0.8/g.odd_single

    for c in ['horse_past_race', 'horse_past_race_dist', 'jockey_past_race', 'jockey_life_win#']:
        df[c] = df[c].astype('int')

    df.sort_values(['race_id', 'standing'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("processing targets...")
    df['target_5'] = (df.standing <= 5).astype('int')

    emb_cols = ['horse_lob', 'horse_sex']
    print(f"create embeddings for {emb_cols}")
    embeddings = create_embedding_(df, emb_cols)

    torch.save({'data': df, 'embeddings': embeddings}, settings.prep_data_path)

    return df, embeddings


def eda(df):
    df['valid'] = 0

    cnt = 0
    months = []
    num_games = []
    for horse_id, gdf in df.groupby('horse_id'):
        m = gdf.month.max() - gdf.month.min()
        g = gdf.shape[0]

        months.append(m)
        num_games.append(g)
        if m >= 12 and g >= 5: # at least 16 months & 10 games
            df.loc[gdf.index, 'valid'] = 1
            cnt += 1

    months = np.array(months)
    num_games = np.array(num_games)
    print(months.min(), months.max(), months.mean())
    print(num_games.min(), num_games.max(), num_games.mean())
    print(df[df.valid==1].shape[0])

    all_valid_races_cnt = 0
    for race_id, gdf in df.groupby('race_id'):
        if np.all(gdf.past_race_dist >= 5):
        #if np.sum(gdf.past_race >= 5)/gdf.shape[0] > 0.8:
            all_valid_races_cnt += 1

    print("num races=", len(df.groupby('race_id')))
    print("num valid races=" , all_valid_races_cnt)

    return df

if __name__ == "__main__":
    prep()

# TODO: filter covid affected races
