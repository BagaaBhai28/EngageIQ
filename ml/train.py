import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from backend.data import ROOT, FEATURES, COUNTS, load_live

def main():
    out = ROOT / 'models'
    out.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / 'data/processed/training_clean.csv').sort_values('Post Month', kind='stable')
    # Keep whole months together; month is the finest reliable chronological key.
    train, test = data[data['Post Month'] <= 9], data[data['Post Month'] > 9]
    numeric = Pipeline([('impute', SimpleImputer()), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([('numeric', numeric, [FEATURES[0]]), ('category', categorical, FEATURES[1:])])
    estimators = {'Median baseline': DummyRegressor(strategy='median'), 'Ridge': Ridge(alpha=10), 'Decision tree': DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42), 'Random forest': RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_leaf=5, random_state=42, n_jobs=-1), 'Gradient boosting': GradientBoostingRegressor(n_estimators=80, max_depth=2, loss='huber', random_state=42)}
    folds = [(np.flatnonzero(train['Post Month'].to_numpy() < m), np.flatnonzero(train['Post Month'].to_numpy() == m)) for m in [6, 7, 8, 9]]
    comparisons, pipelines = [], {}
    for name, estimator in estimators.items():
        for log in ([False] if name == 'Median baseline' else [False, True]):
            label = name + (' (log target)' if log else '')
            reg = TransformedTargetRegressor(regressor=clone(estimator), func=np.log1p, inverse_func=np.expm1) if log else clone(estimator)
            pipe = Pipeline([('preprocessing', clone(pre)), ('regressor', reg)])
            score = -cross_val_score(pipe, train[FEATURES], train['Total Interactions'], cv=folds, scoring='neg_mean_absolute_error').mean()
            pipelines[label] = pipe
            comparisons.append({'name': label, 'cv_mae': round(float(score), 3)})
    comparisons.sort(key=lambda v: v['cv_mae'])
    selected = comparisons[0]['name']
    model = pipelines[selected].fit(train[FEATURES], train['Total Interactions'])
    predicted = np.maximum(0, model.predict(test[FEATURES]))
    y = test['Total Interactions']
    baseline = pipelines['Median baseline'].fit(train[FEATURES], train['Total Interactions']).predict(test[FEATURES])
    metrics = {'mae': float(mean_absolute_error(y, predicted)), 'rmse': float(np.sqrt(mean_squared_error(y, predicted))), 'r2': float(r2_score(y, predicted)), 'baseline_mae': float(mean_absolute_error(y, baseline))}
    joblib.dump(model, out / 'engagement_model.joblib')
    live = load_live()
    scaler = StandardScaler()
    values = scaler.fit_transform(np.log1p(live[COUNTS]))
    candidates = []
    cluster_models = {}
    for k in [2, 3, 4, 5]:
        cluster = KMeans(n_clusters=k, n_init=10, random_state=42).fit(values)
        score = silhouette_score(values, cluster.labels_, sample_size=min(2000, len(values)), random_state=42)
        candidates.append({'k': k, 'silhouette': float(score)})
        cluster_models[k] = cluster
    best = max(candidates, key=lambda v: v['silhouette'])
    cluster = cluster_models[best['k']]
    live['cluster'] = cluster.labels_
    live.to_csv(ROOT / 'data/processed/engagement_clean.csv', index=False)
    joblib.dump({'scaler': scaler, 'model': cluster}, out / 'clustering_model.joblib')
    report = {'selected_model': selected, 'version': '1.0', 'trained_at': datetime.now(timezone.utc).isoformat(), 'sklearn_version': sklearn.__version__, 'training_records': len(train), 'testing_records': len(test), 'split': 'Train months 1–9; untouched test months 10–12 (2014). Validation uses expanding month folds 6–9.', 'features': FEATURES, 'metrics': metrics, 'comparison': comparisons, 'actual_predicted': [{'actual': int(a), 'predicted': round(float(p), 2)} for a, p in zip(y, predicted)], 'clustering': {'selected_k': best['k'], 'candidates': candidates}, 'page_likes_range': [int(train[FEATURES[0]].min()), int(train[FEATURES[0]].max())], 'thresholds': [float(train['Total Interactions'].quantile(.33)), float(train['Total Interactions'].quantile(.67))]}
    (out / 'model_metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['selected_model', 'training_records', 'testing_records', 'metrics', 'clustering']}, indent=2))
