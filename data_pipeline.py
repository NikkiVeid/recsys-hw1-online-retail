"""UCI download, causal time splits and train-only feature construction."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import re
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from scipy import sparse

DATA_URL = 'https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip'
DATA_SHA256 = 'f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b'
VALID_START = pd.Timestamp('2011-09-01')
TEST_START = pd.Timestamp('2011-11-01')


def load_raw():
    root = Path(os.environ.get('RECSYS_DATA_DIR', 'data'))
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'online-retail.zip'
    if not archive.exists():
        tmp = archive.with_suffix('.part')
        urllib.request.urlretrieve(DATA_URL, tmp)
        tmp.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f'UCI archive changed or is incomplete: {digest}')
    # Cached parsed data are tied to the verified archive and excluded from Git.
    cache = root / (digest[:12] + '.pkl')
    if cache.exists():
        return pd.read_pickle(cache)
    with zipfile.ZipFile(archive) as z:
        with z.open('Online Retail.xlsx') as f:
            raw = pd.read_excel(f, engine='openpyxl', dtype={'InvoiceNo': str, 'StockCode': str})
    raw.to_pickle(cache)
    return raw


def clean_data(raw):
    stages = [('Исходные строки', len(raw))]
    df = raw.drop_duplicates().copy()
    stages.append(('Без полных дубликатов', len(df)))
    df = df.dropna(subset=['CustomerID', 'StockCode', 'InvoiceDate'])
    stages.append(('Известны покупатель, товар и время', len(df)))
    df = df[(df.Quantity > 0) & (df.UnitPrice > 0) & ~df.InvoiceNo.str.upper().str.startswith('C')]
    stages.append(('Положительные покупки по ненулевой цене', len(df)))
    # Exclude delivery, bank charges, discounts, manual entries and other service codes.
    df = df[df.StockCode.str.fullmatch(r'\d{5}[A-Za-z]{0,2}')].copy()
    stages.append(('Только товарные коды', len(df)))
    df['CustomerID'] = df.CustomerID.astype(int)
    df['Description'] = df.Description.fillna('UNKNOWN').str.strip().str.upper()
    # One observation = product in one invoice, not units purchased or duplicated lines.
    events = df.groupby(['CustomerID', 'StockCode', 'InvoiceNo'], sort=True, as_index=False).agg(
        InvoiceDate=('InvoiceDate', 'min'), Quantity=('Quantity', 'sum'),
        UnitPrice=('UnitPrice', 'median'), Description=('Description', 'first'), Country=('Country', 'first'))
    events = events.sort_values(['InvoiceDate', 'InvoiceNo', 'StockCode'], kind='stable').reset_index(drop=True)
    stages.append(('События покупатель–товар–чек', len(events)))
    assert 100_000 <= len(events) <= 2_000_000
    assert events.CustomerID.nunique() >= 1000 and events.StockCode.nunique() >= 1000
    return events, pd.DataFrame(stages, columns=['Этап', 'Строки'])


@dataclass
class TrainingData:
    counts: sparse.csr_matrix
    binary: sparse.csr_matrix
    age_days: sparse.csr_matrix
    features: np.ndarray
    users: np.ndarray
    items: np.ndarray
    metadata: pd.DataFrame
    vocabulary: dict
    cutoff: pd.Timestamp


def build_training(events, cutoff):
    assert events.InvoiceDate.max() < cutoff
    users = np.sort(events.CustomerID.unique())
    items = np.sort(events.StockCode.unique())
    user_map = {v: k for k, v in enumerate(users)}
    item_map = {v: k for k, v in enumerate(items)}
    pairs = events.groupby(['CustomerID', 'StockCode']).agg(
        count=('InvoiceNo', 'nunique'), last=('InvoiceDate', 'max')).reset_index()
    u, i = pairs.CustomerID.map(user_map).to_numpy(), pairs.StockCode.map(item_map).to_numpy()
    shape = (len(users), len(items))
    counts = sparse.csr_matrix((pairs['count'].to_numpy(dtype=float), (u, i)), shape=shape)
    ages = (cutoff - pairs['last']).dt.total_seconds().to_numpy() / 86400
    age_days = sparse.csr_matrix((ages, (u, i)), shape=shape)
    counts.sort_indices()
    age_days.sort_indices()
    binary = counts.copy()
    binary.data[:] = 1.
    # Snapshot uses ONLY information observed before the recommendation time.
    metadata = events.sort_values('InvoiceDate', kind='stable').groupby('StockCode').agg(
        Description=('Description', 'last'), UnitPrice=('UnitPrice', 'median')).reindex(items)
    tokens = [sorted(set(re.findall(r'[A-Z]{3,}', text))) for text in metadata.Description]
    frequency = Counter(word for words in tokens for word in words)
    words = sorted(word for word, count in frequency.items() if count >= 3)
    vocab = {word: len(items)+1+j for j, word in enumerate(words)}
    price_offset = len(items)+len(vocab)+1
    # Fixed price bands in GBP; no future quantiles and no test prices.
    price_bucket = np.digitize(metadata.UnitPrice.to_numpy(), [1., 2., 3., 5., 10., 20.])
    width = 2 + max((len(x) for x in tokens), default=0)
    features = np.zeros((len(items), width), dtype=np.int64)
    features[:, 0] = np.arange(1, len(items)+1)
    for row, row_words in enumerate(tokens):
        ids = [vocab[w] for w in row_words if w in vocab]
        features[row, 1:1+len(ids)] = ids
        features[row, -1] = price_offset + price_bucket[row]
    return TrainingData(counts, binary, age_days, features, users, items, metadata, vocab, cutoff)


def evaluation_cohort(future, data):
    """Warm-user, warm-item, novel-item evaluation; same cohort for every method."""
    user_map = {v: k for k, v in enumerate(data.users)}
    item_map = {v: k for k, v in enumerate(data.items)}
    raw_pairs = future[['CustomerID', 'StockCode']].drop_duplicates()
    known_u = raw_pairs.CustomerID.isin(user_map)
    known_i = raw_pairs.StockCode.isin(item_map)
    warm = raw_pairs[known_u & known_i].copy()
    warm['u'] = warm.CustomerID.map(user_map)
    warm['i'] = warm.StockCode.map(item_map)
    seen = np.asarray(data.binary[warm.u.to_numpy(), warm.i.to_numpy()]).ravel() > 0
    novel = warm[~seen]
    truth = {int(u): np.sort(group.i.to_numpy()) for u, group in novel.groupby('u')}
    # Guarantees exactly ten recommendations, without known-item padding.
    truth = {u: x for u, x in truth.items()
             if data.binary.shape[1] - data.binary[u].nnz >= 10}
    audit = {'future_unique_pairs': len(raw_pairs), 'cold_user_pairs': int((~known_u).sum()),
             'cold_item_pairs': int((~known_i).sum()), 'warm_pairs': len(warm),
             'already_seen_pairs': int(seen.sum()), 'novel_relevant_pairs': sum(map(len, truth.values())),
             'evaluated_users': len(truth), 'future_users': future.CustomerID.nunique(),
             'catalog_items': len(data.items)}
    return truth, audit
