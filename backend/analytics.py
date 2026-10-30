import json
from .data import REACTIONS

def records(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso'))

def summarize(data):
    total = data.total_interactions
    content = data.groupby('status_type').agg(posts=('status_id', 'size'), average=('total_interactions', 'mean'), median=('total_interactions', 'median'), comments=('num_comments', 'mean'), shares=('num_shares', 'mean')).reset_index()
    trend = data.groupby('publication_month').agg(interactions=('total_interactions', 'sum'), posts=('status_id', 'size')).reset_index()
    time = data.groupby(['publication_day', 'publication_hour']).agg(average=('total_interactions', 'mean'), posts=('status_id', 'size')).reset_index()
    clusters = []
    if 'cluster' in data:
        clusters = records(data.groupby('cluster').agg(posts=('status_id', 'size'), average=('total_interactions', 'mean'), reactions=('num_reactions', 'mean'), comments=('num_comments', 'mean'), shares=('num_shares', 'mean')).reset_index().sort_values('average'))
    return {'posts': len(data), 'interactions': float(total.sum()), 'average': float(total.mean()) if len(data) else 0, 'median': float(total.median()) if len(data) else 0, 'reactions': float(data.num_reactions.sum()), 'comments': float(data.num_comments.sum()), 'shares': float(data.num_shares.sum()), 'start': str(data.status_published.min().date()) if len(data) else None, 'end': str(data.status_published.max().date()) if len(data) else None, 'content': records(content), 'trend': records(trend), 'timing': records(time), 'reaction_mix': [{'name': c.removeprefix('num_'), 'value': float(data[c].sum())} for c in REACTIONS if c in data], 'clusters': clusters, 'top_posts': records(data.nlargest(8, 'total_interactions')[['status_id', 'status_type', 'status_published', 'num_reactions', 'num_comments', 'num_shares', 'total_interactions']])}
