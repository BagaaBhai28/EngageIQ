import json
import sqlite3
import uuid
from contextlib import contextmanager
from io import BytesIO
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, send_file, session
from werkzeug.exceptions import HTTPException
from .data import ROOT, FEATURES, REACTIONS, clean_live, load_live
from .analytics import summarize

@contextmanager
def connection(path):
    db = sqlite3.connect(path)
    try:
        with db:
            yield db
    finally:
        db.close()

def create_app(config=None):
    app = Flask(__name__, template_folder=str(ROOT / 'templates'), static_folder=str(ROOT / 'static'))
    instance = ROOT / 'instance'
    instance.mkdir(exist_ok=True)
    secret_path = instance / 'secret.key'
    if not secret_path.exists():
        import secrets
        secret_path.write_bytes(secrets.token_bytes(32))
    app.config.update(SECRET_KEY=secret_path.read_bytes(), MAX_CONTENT_LENGTH=8 * 1024 * 1024, DATABASE=str(instance / 'engageiq.sqlite'), UPLOAD_FOLDER=str(instance), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
    if config:
        app.config.update(config)
    uploads = ROOT / app.config['UPLOAD_FOLDER']
    uploads.mkdir(parents=True, exist_ok=True)
    with connection(app.config['DATABASE']) as db:
        db.execute('CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, owner TEXT, created_at TEXT, inputs TEXT, result REAL)')

    def owner():
        if 'owner' not in session:
            session['owner'] = uuid.uuid4().hex
        return session['owner']

    def frame():
        token = request.args.get('dataset', 'builtin')
        if token == 'uploaded':
            path = uploads / (owner() + '.csv')
            if not path.exists():
                raise ValueError('Upload a dataset first.')
            return pd.read_csv(path, parse_dates=['status_published'])
        return load_live()

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/health')
    def health():
        return jsonify(data={'application': 'EngageIQ', 'workspace': str(ROOT)})

    @app.get('/api/analytics')
    def analytics():
        data = frame()
        if request.args.get('type'):
            data = data[data.status_type == request.args['type']]
        for key, op in [('start', 'start'), ('end', 'end')]:
            if request.args.get(key):
                date = pd.to_datetime(request.args[key], errors='raise')
                data = data[data.status_published >= date] if op == 'start' else data[data.status_published < date + pd.Timedelta(days=1)]
        return jsonify(data=summarize(data), source='uploaded' if request.args.get('dataset') == 'uploaded' else 'UCI Facebook Live Sellers')

    @app.get('/api/model')
    def model():
        return jsonify(data=json.loads((ROOT / 'models/model_metrics.json').read_text()))

    @app.post('/api/predict')
    def predict():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError('Provide a JSON object.')
        limits = {'Page total likes': (1, 1000000000), 'Category': (1, 3), 'Post Month': (1, 12), 'Post Weekday': (1, 7), 'Post Hour': (0, 23), 'Paid': (0, 1)}
        inputs = {}
        for key, (low, high) in limits.items():
            value = body.get(key)
            if isinstance(value, bool) or value is None:
                raise ValueError(f'{key} must be an integer between {low} and {high}.')
            try:
                number = float(value)
            except (ValueError, TypeError):
                raise ValueError(f'{key} must be a number.')
            if not np.isfinite(number) or not number.is_integer() or not low <= number <= high:
                raise ValueError(f'{key} must be an integer between {low} and {high}.')
            inputs[key] = int(number)
        if body.get('Type') not in ['Photo', 'Video', 'Status', 'Link']:
            raise ValueError('Choose a supported post type.')
        inputs['Type'] = body['Type']
        model_path = ROOT / 'models/engagement_model.joblib'
        if not hasattr(app, 'prediction_model'):
            app.prediction_model = joblib.load(model_path)
        result = max(0., float(app.prediction_model.predict(pd.DataFrame([inputs], columns=FEATURES))[0]))
        report = json.loads((ROOT / 'models/model_metrics.json').read_text())
        lo, hi = report['page_likes_range']
        warning = 'Page likes are outside the training range; extrapolation is unreliable.' if not lo <= inputs['Page total likes'] <= hi else None
        level = ['Low', 'Medium', 'High'][int(np.searchsorted(report['thresholds'], result))]
        with connection(app.config['DATABASE']) as db:
            db.execute('INSERT INTO predictions(owner,created_at,inputs,result) VALUES(?,?,?,?)', (owner(), datetime.now(timezone.utc).isoformat(), json.dumps(inputs), result))
        return jsonify(data={'interactions': round(result, 1), 'level': level, 'warning': warning, 'model': report['selected_model'], 'test_mae': report['metrics']['mae']})

    @app.get('/api/history')
    def history():
        with connection(app.config['DATABASE']) as db:
            rows = db.execute('SELECT id,created_at,inputs,result FROM predictions WHERE owner=? ORDER BY id DESC LIMIT 100', (owner(),)).fetchall()
        return jsonify(data=[{'id': r[0], 'date': r[1], 'inputs': json.loads(r[2]), 'interactions': round(r[3], 1)} for r in rows])

    @app.post('/api/upload')
    def upload():
        file = request.files.get('file')
        if not file or not file.filename.lower().endswith('.csv'):
            raise ValueError('Choose a CSV file (up to 8 MB).')
        try:
            data = pd.read_csv(file, sep=None, engine='python')
            if len(data) > 100000:
                raise ValueError('Maximum 100,000 rows per upload.')
            clean, report = clean_live(data)
        except (pd.errors.ParserError, UnicodeError):
            raise ValueError('Unable to read this CSV. Use a UTF-8 CSV with column headers.')
        clean.to_csv(uploads / (owner() + '.csv'), index=False)
        return jsonify(data={'quality': report, 'preview': json.loads(clean.head(5).to_json(orient='records', date_format='iso'))})

    @app.get('/api/export')
    def export():
        # Export the same filtered population shown in the dashboard.
        data = frame()
        if request.args.get('type'):
            data = data[data.status_type == request.args['type']]
        for key in ['start', 'end']:
            if request.args.get(key):
                date = pd.to_datetime(request.args[key], errors='raise')
                data = data[data.status_published >= date] if key == 'start' else data[data.status_published < date + pd.Timedelta(days=1)]
        return send_file(BytesIO(data.to_csv(index=False).encode('utf-8-sig')), mimetype='text/csv', as_attachment=True, download_name='engageiq-posts.csv')

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(Exception)
    def unexpected(error):
        app.logger.exception('Request failed')
        return jsonify(error='The request could not be completed. Check local datasets and model artifacts.'), 500

    return app
