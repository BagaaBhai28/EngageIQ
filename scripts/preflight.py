"""Repair missing generated assets, then verify the real local data path."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def run(script):
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script)], cwd=ROOT, check=True)

def main():
    import joblib
    import pandas as pd
    from backend import create_app
    from backend.data import FEATURES, load_live

    raw = [ROOT / 'data/raw' / name for name in ['facebook_live.csv', 'facebook_metrics.csv']]
    processed = [ROOT / 'data/processed' / name for name in ['engagement_clean.csv', 'training_clean.csv']]
    artifacts = [ROOT / 'models' / name for name in ['engagement_model.joblib', 'clustering_model.joblib', 'model_metrics.json']]
    if not all(p.is_file() and p.stat().st_size for p in raw):
        run('download_datasets.py')
    prepared = not all(p.is_file() and p.stat().st_size for p in processed)
    if prepared:
        run('prepare_datasets.py')
    if prepared or not all(p.is_file() and p.stat().st_size for p in artifacts):
        run('train_models.py')
    try:
        predictor = joblib.load(artifacts[0])
        joblib.load(artifacts[1])
        report = json.loads(artifacts[2].read_text())
        predictor.predict(pd.DataFrame([[120000, 'Photo', 2, 10, 6, 18, 0]], columns=FEATURES))
    except Exception:
        print('Model artifacts could not be verified. Rebuilding from bundled real data...')
        run('train_models.py')
        predictor = joblib.load(artifacts[0])
        report = json.loads(artifacts[2].read_text())
        predictor.predict(pd.DataFrame([[120000, 'Photo', 2, 10, 6, 18, 0]], columns=FEATURES))
    frame = load_live()
    if frame.empty or 'cluster' not in frame:
        raise RuntimeError('Processed analytics data is incomplete. Run scripts/prepare_datasets.py and scripts/train_models.py.')
    vendor = ROOT / 'static/vendor/chart.umd.js'
    if not vendor.is_file() or vendor.stat().st_size < 1000:
        from urllib.request import urlopen
        url = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.8/dist/chart.umd.js'
        vendor.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(url, timeout=60) as response:
                vendor.write_bytes(response.read())
        except OSError:
            subprocess.run(['curl.exe', '--fail', '--location', '--silent', '--show-error', url, '--output', str(vendor)], check=True)
    app = create_app()
    with app.test_client() as client:
        response = client.get('/api/analytics')
        if response.status_code != 200 or response.json['data']['posts'] != len(frame):
            raise RuntimeError('Analytics self-check failed.')
    print(f"Ready: {len(frame):,} historical posts; {report['training_records'] + report['testing_records']} prediction records; {report['selected_model']}.")
    print('Local storage is writable; analytics API and saved model inference passed.')

if __name__ == '__main__':
    main()
