"""Only the audited, pre-2024 snapshot is allowed; never load a mixed-year file."""
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

APPROVED_SHA256 = '23972d31dd5ff90b46da05703e1b75dbfbbbbfde6efcd26b6b9ad24ee3b65503'

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_data(path):
    # This hash was registered by the original pre2024 artifact manifest.
    # Refuse unknown snapshots BEFORE deserializing market observations.
    digest = sha256(path)
    if digest != APPROVED_SHA256:
        raise ValueError('Unapproved snapshot: require independently audited pre-2024 file hash')
    d = pd.read_feather(path, columns=['date','open','high','low','close','volume'])
    d['date'] = pd.to_datetime(d.date, utc=True)
    expected = pd.date_range('2020-01-01', '2023-12-31 23:45', freq='15min', tz='UTC')
    duplicates = int(d.date.duplicated().sum())
    missing = len(expected.difference(pd.DatetimeIndex(d.date)))
    if duplicates or missing or not np.array_equal(d.date.array, expected.array):
        raise ValueError('Snapshot must contain exactly contiguous 2020-2023 candles, ordered and unique')
    values = d[['open','high','low','close']].to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Invalid price')
    if not ((d.high >= d[['open','close','low']].max(axis=1)) &
            (d.low <= d[['open','close','high']].min(axis=1))).all():
        raise ValueError('Inconsistent OHLC')
    if not np.isfinite(d.volume).all() or (d.volume < 0).any():
        raise ValueError('Invalid volume')
    return d, dict(path=str(Path(path).resolve()), sha256=digest, bars=len(d),
                  first=str(d.date.iloc[0]), last=str(d.date.iloc[-1]),
                  missing=missing, duplicates=duplicates, timezone='UTC',
                  candle_timestamp='open; close is available 15 minutes later',
                  zero_volume_bars=int(d.volume.eq(0).sum()))
