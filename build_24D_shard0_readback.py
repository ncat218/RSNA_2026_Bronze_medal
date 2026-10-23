"""Build the CPU-only, streaming readback audit for the saved 24C shard-0 output."""

import json
from pathlib import Path


OUT = Path(__file__).with_name("24D_Competition_Only_Cache_Shard0_Readback_CPU.ipynb")

markdown = """# 24D — shard-0 cache readback QA (CPU only)

Attach the **saved Output** of `24C_Competition_Only_Cache_Shard0_CPU`, containing `24C_shard0_receipt.json` and `rsna_24c_shard0/manifest.csv` plus all eleven `chunk_*.zip` files. No competition data, labels, GPU, internet, or other notebook output is needed. Set Accelerator **None**, Internet **Off**, then **Save & Run All**.

This audit streams every 24C NPZ from the read-only attached ZIPs, checks CRC, byte count and SHA-256, and decodes a bounded sample (first/last in every chunk plus unusual slice counts) to check shape, dtype and mask. It does **not** train a model or create a new cache. Send `24D_shard0_qa.json` back. Do not start shards 1–3 unless the JSON says `PASS`.
"""

code = r'''from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from time import monotonic
import csv, hashlib, io, json, re, zipfile
import numpy as np

START = monotonic()
INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working/24D_shard0_qa.json')
EXPECTED = {
    'shard_id': 0,
    'n_shards': 4,
    'assigned_studies': 1088,
    'assigned_uid_sha256': '1579721e4ae48c7c0c24cd5e03c4c567cb805ef659efcfdcee195a356c7ef0c9',
    'train_csv_sha256': '8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33',
    'train_series_csv_sha256': '573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3',
}
report = {
    'experiment': '24D_shard0_readback_qa',
    'time_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'FAIL',
    'cuda_used': False,
    'training_started': False,
    'errors': [],
    'warnings': [],
    'archives_checked': 0,
    'npz_streamed_and_hashed': 0,
    'npz_sample_decoded': 0,
    'bytes_streamed': 0,
}

def problem(message):
    if len(report['errors']) < 30:
        report['errors'].append(str(message))

def audit():
    receipts = sorted(INPUT.rglob('24C_shard0_receipt.json'))
    report['candidate_receipts'] = [str(p) for p in receipts]
    if len(receipts) != 1:
        problem(f'Expected exactly one saved 24C shard-0 Output attachment; found {len(receipts)}')
        return
    receipt_path = receipts[0]
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    cache_dir = receipt_path.parent / 'rsna_24c_shard0'
    manifest_path = cache_dir / 'manifest.csv'
    report['attachment_root'] = str(receipt_path.parent)
    report['cache_dir'] = str(cache_dir)
    report['source_status'] = receipt.get('status')
    if receipt.get('status') != 'COMPLETE' or receipt.get('errors') or receipt.get('warnings'):
        problem('24C source receipt is not clean and COMPLETE')
    for key, expected in EXPECTED.items():
        if receipt.get(key) != expected:
            problem(f'24C receipt {key} mismatch: {receipt.get(key)!r}')
    if not manifest_path.is_file():
        problem('Missing manifest.csv alongside saved ZIP chunks')
        return
    with manifest_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = ['StudyInstanceUID', 'archive', 'entry', 'npz_bytes', 'npz_sha256',
                    'valid_slices', 'decoded_slices', 'failed_slices']
        if reader.fieldnames != required:
            problem(f'Manifest columns differ: {reader.fieldnames!r}')
            return
        rows = list(reader)
    report['manifest_rows'] = len(rows)
    if len(rows) != EXPECTED['assigned_studies'] or receipt.get('built_studies') != len(rows):
        problem('Manifest length differs from shard assignment or source receipt')
    uids = [r['StudyInstanceUID'] for r in rows]
    if len(set(uids)) != len(uids) or uids != sorted(uids):
        problem('Manifest UIDs are duplicated or not sorted')
    digest = hashlib.sha256('\n'.join(uids).encode()).hexdigest()
    report['recomputed_assigned_uid_sha256'] = digest
    if digest != EXPECTED['assigned_uid_sha256']:
        problem('Manifest assigned-ID SHA256 mismatch')

    by_archive = defaultdict(list)
    for r in rows:
        uid = r['StudyInstanceUID']
        if re.fullmatch(r'[0-9.]+', uid) is None:
            problem(f'Unexpected study UID syntax: {uid}')
        if r['entry'] != f'study_{uid}.npz':
            problem(f'Entry name mismatch: {uid}')
        if re.fullmatch(r'[0-9a-f]{64}', r['npz_sha256']) is None:
            problem(f'Invalid NPZ SHA256: {uid}')
        try:
            size = int(r['npz_bytes'])
            valid = int(r['valid_slices'])
            decoded = int(r['decoded_slices'])
            failed = int(r['failed_slices'])
            if size <= 0 or not 40 <= valid <= 96 or decoded < valid or failed < 0:
                problem(f'Implausible study metadata: {uid}')
        except ValueError:
            problem(f'Non-integer study metadata: {uid}')
        by_archive[r['archive']].append(r)
    expected_archives = [f'chunk_{i:02d}.zip' for i in range(11)]
    if sorted(by_archive) != expected_archives:
        problem(f'Expected 11 ordered chunks; found {sorted(by_archive)}')
        return
    for i, name in enumerate(expected_archives):
        expected_count = 88 if i == 10 else 100
        if len(by_archive[name]) != expected_count:
            problem(f'{name} has {len(by_archive[name])} manifest entries, expected {expected_count}')
    published_sizes = {item['name']: int(item['bytes']) for item in receipt.get('zip_files', [])}
    if sorted(published_sizes) != expected_archives:
        problem('ZIP size inventory in source receipt differs from manifest')

    for name in expected_archives:
        archive_path = cache_dir / name
        if not archive_path.is_file():
            problem(f'Missing archive: {name}')
            continue
        if archive_path.stat().st_size != published_sizes.get(name):
            problem(f'ZIP byte count differs from source receipt: {name}')
        group = by_archive[name]
        sample_uids = {group[0]['StudyInstanceUID'], group[-1]['StudyInstanceUID']}
        sample_uids.update(r['StudyInstanceUID'] for r in group if int(r['valid_slices']) not in (74, 84, 96))
        try:
            with zipfile.ZipFile(archive_path) as z:
                names = z.namelist()
                wanted = [r['entry'] for r in group]
                if len(names) != len(set(names)) or set(names) != set(wanted):
                    problem(f'ZIP entries do not match manifest: {name}')
                    continue
                report['archives_checked'] += 1
                for r in group:
                    uid = r['StudyInstanceUID']
                    info = z.getinfo(r['entry'])
                    if info.file_size != int(r['npz_bytes']):
                        problem(f'NPZ byte count differs: {uid}')
                    digest = hashlib.sha256()
                    total = 0
                    sample = io.BytesIO() if uid in sample_uids else None
                    with z.open(r['entry']) as stream:
                        while True:
                            block = stream.read(1024 * 1024)
                            if not block:
                                break
                            digest.update(block)
                            total += len(block)
                            if sample is not None:
                                sample.write(block)
                    report['npz_streamed_and_hashed'] += 1
                    report['bytes_streamed'] += total
                    if total != int(r['npz_bytes']) or digest.hexdigest() != r['npz_sha256']:
                        problem(f'NPZ SHA256/size mismatch: {uid}')
                    if sample is not None:
                        sample.seek(0)
                        with np.load(sample, allow_pickle=False) as npz:
                            if set(npz.files) != {'images', 'mask'}:
                                problem(f'Unexpected NPZ fields: {uid}')
                                continue
                            images, mask = npz['images'], npz['mask']
                        if images.shape != (96, 384, 384) or images.dtype != np.uint8:
                            problem(f'Invalid image shape/dtype: {uid}')
                        if mask.shape != (96,) or mask.dtype != np.uint8:
                            problem(f'Invalid mask shape/dtype: {uid}')
                        if not np.array_equal(mask, np.any(images != 0, axis=(1, 2)).astype(np.uint8)):
                            problem(f'Mask and image nonzero slices differ: {uid}')
                        if int(mask.sum()) != int(r['valid_slices']):
                            problem(f'Mask and manifest valid-slice count differ: {uid}')
                        report['npz_sample_decoded'] += 1
                        del images, mask, sample
        except Exception as e:
            problem(f'ZIP/NPZ readback failed in {name}: {type(e).__name__}: {str(e)[:180]}')
    if report['npz_streamed_and_hashed'] != len(rows):
        problem('Not every manifest NPZ was streamed and hashed')
    expected_total = sum(int(r['npz_bytes']) for r in rows)
    if report['bytes_streamed'] != expected_total:
        problem('Total streamed bytes differ from manifest')
    report['manifest_npz_bytes'] = expected_total
    report['valid_slice_counts'] = dict(sorted(Counter(int(r['valid_slices']) for r in rows).items()))
    report['decoded_minus_valid_one_studies'] = sum(
        int(r['decoded_slices']) - int(r['valid_slices']) == 1 for r in rows)
    report['failed_slice_total'] = sum(int(r['failed_slices']) for r in rows)
    if not report['errors']:
        report['status'] = 'PASS'

try:
    audit()
except Exception as e:
    problem(f'Uncaught audit failure: {type(e).__name__}: {str(e)[:240]}')
report['elapsed_seconds'] = round(monotonic() - START, 2)
OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print('QA status:', report['status'])
print('Archives:', report['archives_checked'], 'NPZ hashed:', report['npz_streamed_and_hashed'],
      'sample decoded:', report['npz_sample_decoded'], 'elapsed seconds:', report['elapsed_seconds'])
print('Report:', OUTPUT)
if report['errors']:
    print('Errors:', report['errors'][:5])
'''

compile(code, str(OUT), "exec")
notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": code.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUT)
