"""Load only six raw OHLCV columns from the exact approved snapshot."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

EXPECTED_SHA256='23972d31dd5ff90b46da05703e1b75dbfbbbbfde6efcd26b6b9ad24ee3b65503'
RAW_COLUMNS=['date','open','high','low','close','volume']

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_training(path):
    digest=sha256(path)
    if digest!=EXPECTED_SHA256:raise ValueError('SHA256 mismatch: STOP before deserializing')
    raw=pd.read_feather(path,columns=RAW_COLUMNS)
    raw['date']=pd.to_datetime(raw.date,utc=True)
    expected=pd.date_range('2020-01-01','2023-12-31 23:45',freq='15min',tz='UTC')
    if len(raw)!=140256 or not np.array_equal(raw.date.array,expected.array):
        raise ValueError('Raw snapshot is not unique, ordered, contiguous 2020-2023 data')
    # Validate training observations only; do not calculate holdout price features or outcomes.
    d=raw.loc[raw.date < pd.Timestamp('2022-01-01',tz='UTC'),RAW_COLUMNS].copy().reset_index(drop=True)
    total_zero=int(raw.volume.eq(0).sum()) # raw-data QC only, never holdout performance
    if total_zero!=97:raise ValueError('Unexpected zero-volume count')
    p=d[['open','high','low','close']].to_numpy()
    if not np.isfinite(p).all() or (p<=0).any():raise ValueError('Invalid training prices')
    if not ((d.high>=d[['open','close','low']].max(axis=1))&(d.low<=d[['open','close','high']].min(axis=1))).all():
        raise ValueError('Inconsistent OHLC')
    if not np.isfinite(d.volume).all() or (d.volume<0).any():raise ValueError('Invalid volume')
    meta=dict(path=str(Path(path).resolve()),sha256=digest,raw_bars=len(raw),raw_first=str(raw.date.iloc[0]),
        raw_last=str(raw.date.iloc[-1]),missing=0,duplicates=0,raw_zero_volume=total_zero,
        training_bars=len(d),training_first=str(d.date.iloc[0]),training_last=str(d.date.iloc[-1]),
        training_zero_volume=int(d.volume.eq(0).sum()),columns_read=RAW_COLUMNS,
        holdout_usage='snapshot structural QC only; 2022/2023 removed before indicators/execution/metrics',
        policy='zero-volume records retained unchanged; no forward fill or price imputation')
    return d,meta
