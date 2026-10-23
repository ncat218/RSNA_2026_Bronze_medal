"""Fetch public source for inspection only; never execute downloaded code."""
import hashlib, json, urllib.request
from pathlib import Path

out = Path(__file__).parent / 'research_38_s6'
out.mkdir(exist_ok=True)
url = 'https://www.kaggle.com/api/v1/kernels/pull/nartaa/rsna-knee-s6fall-solo'
with urllib.request.urlopen(url, timeout=60) as response:
    data = json.load(response)
source = data['blob']['source']
path = out / 'public_s6_source.py'
if path.exists() and path.read_text(encoding='utf-8') != source:
    raise RuntimeError('Source changed; preserve existing audit and inspect separately')
path.write_text(source, encoding='utf-8')
(out / 'source_metadata.json').write_text(json.dumps(data['metadata'], indent=2), encoding='utf-8')
print('source sha256:', hashlib.sha256(source.encode()).hexdigest())
for i, line in enumerate(source.splitlines(), 1):
    if len(line) < 1500:
        print(f'{i}: {line}')
