"""Start the local app without colliding with another project; open only when ready."""
import argparse
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import create_app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    app = create_app()
    server = None
    for port in range(5055, 5066):
        url = f'http://127.0.0.1:{port}'
        try:
            with urlopen(url + '/api/health', timeout=1) as response:
                health = json.load(response)['data']
            if health.get('application') == 'EngageIQ' and health.get('workspace') == str(ROOT):
                print(f'EngageIQ is already running: {url}', flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
                return
        except Exception:
            pass
        try:
            # make_server binds now, avoiding a probe/start race.
            server = make_server('127.0.0.1', port, app, threaded=True)
            break
        except (OSError, SystemExit):
            continue
    if server is None:
        raise RuntimeError('Ports 5055-5065 are occupied. Close an unused local server and rerun.')
    print(f'\nEngageIQ is running at {url}\nPress Ctrl+C in this terminal to stop.\n', flush=True)
    if not args.no_browser:
        def open_when_ready():
            for _ in range(30):
                try:
                    with urlopen(url + '/api/health', timeout=1) as response:
                        if response.status == 200:
                            webbrowser.open(url)
                            return
                except OSError:
                    time.sleep(.3)
        threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nEngageIQ stopped. Saved data remains available.')
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
