"""Non-training sanity checks: run `python verify_assets.py` before deployment."""
from pathlib import Path
import hashlib, json, zipfile
import pandas as pd

BASE = Path(__file__).resolve().parent
MANIFEST = json.loads((BASE/'assets/data/asset_manifest.json').read_text())


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()

for category,folder in [('model_files','models'),('data_files','data')]:
    for filename,expected in MANIFEST[category].items():
        p=BASE/'assets'/folder/filename
        assert p.is_file(), f'Missing file: {p}'
        assert digest(p)==expected, f'Hash mismatch (possibly stale): {p}'
        if p.suffix=='.keras':
            with zipfile.ZipFile(p) as z:
                assert {'metadata.json','config.json','model.weights.h5'}<=set(z.namelist())

r=pd.read_csv(BASE/'assets/data/results.csv')
assert len(r)==21 and (r.TN+r.FP==60).all() and (r.TP+r.FN==20).all()
assert abs(r.loc[(r.model=='EfficientNetB0') & (r.strategy=='baseline'),'mcc'].iloc[0]-0.7216878364870322)<1e-6
assert abs(r.loc[(r.model=='EfficientNetB0') & (r.strategy=='class weights'),'recall'].iloc[0]-0.9)<1e-6
assert len(pd.read_csv(BASE/'assets/data/repeated_cv.csv'))==12
assert len(pd.read_csv(BASE/'assets/data/clarity_features.csv'))==400
assert not (BASE/'assets/data/test_scores.csv').exists(), 'Stale per-image scores must not be deployed.'
print('PASSED: hashes, two Keras archives, 21 result rows, 12 CV rows, 400 clarity rows.')
print('Note: this does not replace actual TensorFlow inference or a Streamlit live test.')
