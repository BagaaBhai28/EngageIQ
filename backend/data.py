from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ['num_reactions', 'num_comments', 'num_shares']
REACTIONS = ['num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']
FEATURES = ['Page total likes', 'Type', 'Category', 'Post Month', 'Post Weekday', 'Post Hour', 'Paid']

def clean_live(frame):
    frame = frame.copy().dropna(axis=1, how='all')
    aliases = {'post_id': 'status_id', 'post_type': 'status_type', 'post_date': 'status_published', 'reactions': 'num_reactions', 'likes': 'num_likes', 'comments': 'num_comments', 'shares': 'num_shares'}
    frame = frame.rename(columns={k: v for k, v in aliases.items() if v not in frame})
    if 'num_reactions' not in frame and 'num_likes' in frame:
        frame['num_reactions'] = frame['num_likes']
    required = ['status_type', 'status_published', *COUNTS]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    original = len(frame)
    frame['status_type'] = frame.status_type.astype(str).str.strip().str.lower().replace({'image': 'photo'})
    frame['status_published'] = pd.to_datetime(frame.status_published, errors='coerce', format='mixed')
    for c in COUNTS + [c for c in REACTIONS if c in frame]:
        frame[c] = pd.to_numeric(frame[c], errors='coerce')
    valid = frame[COUNTS].notna().all(axis=1) & np.isfinite(frame[COUNTS]).all(axis=1) & (frame[COUNTS] >= 0).all(axis=1)
    frame = frame[valid & frame.status_published.notna() & frame.status_type.isin(['photo', 'video', 'status', 'link'])]
    frame = frame.drop_duplicates()
    if 'status_id' in frame:
        frame = frame.drop_duplicates('status_id')
    else:
        frame['status_id'] = ['upload-' + str(i) for i in range(len(frame))]
    if frame.empty:
        raise ValueError('No valid posts remain after validation.')
    frame['total_interactions'] = frame[COUNTS].sum(axis=1)
    frame['publication_hour'] = frame.status_published.dt.hour
    frame['publication_day'] = frame.status_published.dt.dayofweek
    frame['publication_month'] = frame.status_published.dt.to_period('M').astype(str)
    report = {'input_rows': original, 'valid_rows': len(frame), 'removed_rows': original - len(frame)}
    return frame, report

def prepare():
    import json
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    live, report = clean_live(pd.read_csv(ROOT / 'data/raw/facebook_live.csv'))
    live.to_csv(out / 'engagement_clean.csv', index=False)
    train = pd.read_csv(ROOT / 'data/raw/facebook_metrics.csv', sep=';')
    train = train.drop_duplicates().dropna(subset=['Total Interactions'])
    train.to_csv(out / 'training_clean.csv', index=False)
    report['training_rows'] = len(train)
    report['target_component_mismatches'] = int((train[['like', 'comment', 'share']].sum(axis=1) != train['Total Interactions']).sum())
    (out / 'quality_report.json').write_text(json.dumps(report, indent=2))
    print(report)

def load_live():
    return pd.read_csv(ROOT / 'data/processed/engagement_clean.csv', parse_dates=['status_published'])
