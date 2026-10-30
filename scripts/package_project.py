"""Create a portable source bundle without environments or private runtime data."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
target = root / 'EngageIQ.zip'
excluded = {'.venv', '.git', '__pycache__', '.pytest_cache', 'instance', 'tmp'}
with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
    for file in root.rglob('*'):
        if file.is_file() and file != target and not any(part in excluded for part in file.relative_to(root).parts):
            archive.write(file, Path('EngageIQ') / file.relative_to(root))
print(target)
