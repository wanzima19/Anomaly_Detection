import os
import zipfile

BASE = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE, 'output')
DEST = os.path.join(OUT_DIR, 'models_package.zip')

os.makedirs(OUT_DIR, exist_ok=True)

with zipfile.ZipFile(DEST, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    if os.path.exists(OUT_DIR):
        for root, _, files in os.walk(OUT_DIR):
            for f in files:
                if f == os.path.basename(DEST):
                    continue
                full = os.path.join(root, f)
                arcname = os.path.relpath(full, OUT_DIR)
                zf.write(full, arcname)

print(f'Packaged models into: {DEST}')
