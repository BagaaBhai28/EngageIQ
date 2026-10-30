"""Download original UCI archives, preserving the real source CSVs."""
from pathlib import Path
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import hashlib
import json
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    output = ROOT / 'data/raw'
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for ident, slug, filename in [(488, 'facebook+live+sellers+in+thailand', 'facebook_live.csv'), (368, 'facebook+metrics', 'facebook_metrics.csv')]:
        url = f'https://archive.ics.uci.edu/static/public/{ident}/{slug}.zip'
        try:
            with urlopen(url, timeout=60) as response:
                archive = response.read()
        except OSError:
            # Windows curl uses the OS certificate store; preserve TLS verification.
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / 'dataset.zip'
                subprocess.run(['curl.exe', '--fail', '--location', '--silent', '--show-error', url, '--output', str(target)], check=True)
                archive = target.read_bytes()
        with ZipFile(BytesIO(archive)) as z:
            member = next(n for n in z.namelist() if n.lower().endswith('.csv'))
            data = z.read(member)
        (output / filename).write_bytes(data)
        sources[filename] = {'url': url, 'original_file': member, 'sha256': hashlib.sha256(data).hexdigest(), 'license': 'CC BY 4.0'}
    (output / 'sources.json').write_text(json.dumps(sources, indent=2))
    print('Downloaded both original UCI datasets.')

if __name__ == '__main__':
    main()
