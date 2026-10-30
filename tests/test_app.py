from io import BytesIO
import sqlite3
import pandas as pd
import pytest
from backend import create_app, connection
from backend.data import clean_live, load_live

@pytest.fixture
def app(tmp_path):
    return create_app({'TESTING': True, 'DATABASE': str(tmp_path / 'test.sqlite'), 'UPLOAD_FOLDER':str(tmp_path / 'uploads'), 'SECRET_KEY': 'test-only'})

@pytest.fixture
def client(app):
    return app.test_client()

def payload():
    return {'Page total likes': 120000, 'Type': 'Photo', 'Category': 2, 'Post Month': 10, 'Post Weekday': 5, 'Post Hour': 18, 'Paid': 0}

def test_analytics_integrity(client):
    data = client.get('/api/analytics').json['data']
    frame = load_live()
    assert data['posts'] == len(frame)
    assert data['interactions'] == frame[['num_reactions', 'num_comments', 'num_shares']].sum().sum()
    assert data['interactions'] == data['reactions'] + data['comments'] + data['shares']
    photo = client.get('/api/analytics?type=photo').json['data']
    assert photo['posts'] == int((frame.status_type == 'photo').sum())
    assert client.get('/api/analytics?start=2030-01-01').json['data']['posts'] == 0

def test_predictions_and_persistence(app, client):
    a = client.post('/api/predict', json=payload())
    b = client.post('/api/predict', json=payload())
    assert a.status_code == 200 and a.json['data']['interactions'] == b.json['data']['interactions']
    assert len(client.get('/api/history').json['data']) == 2
    with connection(app.config['DATABASE']) as db:
        assert db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] == 2
    # A new app instance reads the same persistent DB with the original browser cookie.
    restarted = create_app({'TESTING':True, 'DATABASE':app.config['DATABASE'], 'SECRET_KEY':'test-only'}).test_client()
    restarted.set_cookie('session', client.get_cookie('session').value)
    assert len(restarted.get('/api/history').json['data']) == 2
    assert app.test_client().get('/api/history').json['data'] == []

@pytest.mark.parametrize('key,value', [('Post Hour',24), ('Paid',-1), ('Category',1.5), ('Type','Reel'), ('Page total likes','NaN'), ('Post Month',None)])
def test_invalid_prediction(client,key,value):
    body=payload(); body[key]=value
    assert client.post('/api/predict',json=body).status_code == 400

def test_upload_and_export(client):
    csv=b'post_id,post_type,post_date,likes,comments,shares\na,photo,2020-01-01,10,2,3\nb,video,2020-01-02,20,4,6\nc,video,invalid,20,4,6\n'
    result=client.post('/api/upload',data={'file':(BytesIO(csv),'posts.csv')})
    assert result.status_code == 200
    assert result.json['data']['quality'] == {'input_rows':3,'valid_rows':2,'removed_rows':1}
    d=client.get('/api/analytics?dataset=uploaded&type=photo').json['data']
    assert d['interactions'] == 15 and d['posts'] == 1
    export=client.get('/api/export?dataset=uploaded&type=photo')
    assert len(pd.read_csv(BytesIO(export.data))) == 1
    assert client.post('/api/upload',data={'file':(BytesIO(b'wrong,header\n1,2'),'bad.csv')}).status_code == 400
    assert client.post('/api/upload',data={'file':(BytesIO(b'x'),'bad.exe')}).status_code == 400

def test_cleaning_preserves_real_outlier():
    f=pd.DataFrame({'status_type':['photo','photo','photo'], 'status_published':['2020-01-01']*3,'num_reactions':[1000000,-1,5], 'num_comments':[10,0,2], 'num_shares':[2,0,3]})
    clean,report=clean_live(f)
    assert len(clean)==2 and clean.total_interactions.max()==1000012
    assert report['removed_rows']==1

def test_model_provenance(client):
    d=client.get('/api/model').json['data']
    assert d['training_records']+d['testing_records']==500
    assert len(d['features'])==7
    assert len(d['actual_predicted'])==d['testing_records']
    assert 'like' not in d['features']

def test_pages_and_errors(client):
    assert client.get('/').status_code==200
    assert client.get('/static/vendor/chart.umd.js').status_code==200
    assert client.get('/api/missing').status_code==404
    assert client.post('/api/predict',json=[]).status_code==400
