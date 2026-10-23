"""Download one small pinned public source for inspection; never execute it."""
import hashlib, io, tarfile, urllib.request
from pathlib import Path

url='https://www.kaggle.com/api/v1/datasets/download/nartaa/rsna-knee-s6fall-weights?datasetVersionNumber=5&fileName=s6_extras.bin'
data=urllib.request.urlopen(url,timeout=60).read()
assert data.startswith(b'K1PAYLOAD1\n')
with tarfile.open(fileobj=io.BytesIO(data.split(b'\n',1)[1]),mode='r:') as tar:
    source=tar.extractfile('heliosli/compact_infer.py').read()
assert hashlib.sha256(source).hexdigest()=='34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a'
path=Path(__file__).parent/'research_38_s6'/'compact_infer_s6_original.py'
if path.exists():assert path.read_bytes()==source
else:path.write_bytes(source)
print('Pinned source saved for inspection, not executed:',path)
