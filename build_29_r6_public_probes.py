"""Package the supplied R6/mirror public source and two single-factor ablations."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'Kaggle public' / '0-9576-latest-public-dataset-score-in-one-click.ipynb'
ref = json.loads(SOURCE.read_text(encoding='utf-8'))
old = json.loads((ROOT / '25A_Raptor_R7_Public0956_Reproduction.ipynb').read_text(encoding='utf-8'))
s = lambda i: ''.join(ref['cells'][i]['source'])
old_s = lambda i: ''.join(old['cells'][i]['source'])

def cell(text, markdown=False):
    if markdown:
        return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}
    checked = text.split('\n', 1)[1] if text.startswith('%%writefile ') else text
    compile(checked, '<29 generated cell>', 'exec')
    return {'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(keepends=True),
            'execution_count': None, 'outputs': []}

core_hashes = {str(i): hashlib.sha256(s(i).encode()).hexdigest() for i in range(2, 9)}
variants = [
    ('29A', True, True, 'R6_TTA_Raptor384_224_Reproduction'),
    ('29B', True, False, 'R6_TTA_Raptor384_No224'),
    ('29C', False, True, 'R6_Plain_Raptor384_224_NoTTA'),
]
created = []
for tag, flip, use224, suffix in variants:
    title = f'{tag} — R6 {"mirror TTA" if flip else "plain"} + Raptor-384' + (' + Raptor-224' if use224 else '')
    description = ('Exact scoring-code reproduction of the supplied public notebook; the filename claims 0.9576, '
                   'which must be checked by a live submission on this account.' if tag == '29A' else
                   'One-factor ablation relative to 29A: ' + ('omit Raptor-224 and give its share to Raptor-384.' if not use224 else 'disable R6 mirror TTA, retaining its plain prediction.'))
    intro = f'''# {title}

{description}

This source uses **R6 bundle fe055406fad6a12f**, which differs from the R7 bundle 8c7b666e27dc0e8f in our confirmed 0.956 base. The public worker implementation is the same; the learned checkpoint bundle and auxiliary stackers differ. R6 mirror uses anatomical transforms and medial/lateral output re-indexing, with MCL from the plain pass.

Effective rank weights: R6 50%, Raptor-384 {32.5 if use224 else 50}%, Raptor-224 {17.5 if use224 else 0}%. Weights are shared across all twelve findings. Original Raptor crop/mirror and window policies are retained. This probe does not depend on outputs of 28 or another 29 notebook, and needs no training cache or model training.

**Attach four inputs:** RSNA Knee competition; [publication Raptor weights](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007); [R6 frozen fe055406 — VERSION 1 ONLY](https://www.kaggle.com/datasets/heliosli/rsna-knee-r6-frozen-fe055406/versions/1); [OrthoFoundation code/anatomy](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004). **GPU T4x2, Internet Off, Save & Run All.** IMPORTANT: the R6-named dataset's VERSION 2 replaced its contents with R7 under the same filenames. Select VERSION 1 explicitly; the latest version is NOT compatible with this experiment. Model identity is checked before the expensive stages.

**V1-PINNED IMPORT:** This file explicitly pins R6 datasetVersion ID **20611930 (version 1)** in Kaggle metadata. Import into a NEW Kaggle notebook so the old draft's version-2 input is not retained. Importing code into an existing draft may retain its attached inputs; in that case change the R6 input to version 1. Notebook runtime cannot change Kaggle attachments with Internet Off. No model files are embedded in this notebook.

After the saved run, inspect `{tag}_run_audit.json`: submit once if it is COMPLETE and the mirror/224 configuration matches this title. Send the Public score and Output ZIP. The reference's saved runtime (~7.4 minutes) covers only three visible studies; hidden runtime remains unmeasured here. The three notebooks are independent; their results can be compared after all are submitted. 29A is the reproduction anchor. Do not adjust blend weights in response to these scores.

'''
    intro += s(1)
    pre = old_s(2).replace('rsna-knee-compact-r7cnxfsss', 'rsna-knee-r6-frozen-fe055406').replace(
        '8c7b666e27dc0e8f', 'fe055406fad6a12f').replace('R7 bundle_id', 'R6 bundle_id').replace(
        "'25_preflight.json'", f"'{tag}_preflight.json'")
    pre = pre.replace("'r7_bundle'", "'r6_bundle'").replace("'r7_bundle_id'", "'r6_bundle_id'")
    old_guard = "if r7_manifest.get('bundle_id') != 'fe055406fad6a12f':\n    raise RuntimeError('R6 bundle_id differs from the tested public notebook')"
    assert old_guard in pre
    pre = pre.replace(old_guard, f'''import hashlib
expected_manifest_sha256 = '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648'
observed_manifest_sha256 = hashlib.sha256((r7 / 'manifest.json').read_bytes()).hexdigest()
observed_bundle_id = r7_manifest.get('bundle_id')
identity = {{'status': 'CHECKING', 'path': str(r7),
            'expected_bundle_id': 'fe055406fad6a12f', 'observed_bundle_id': observed_bundle_id,
            'expected_manifest_sha256': expected_manifest_sha256,
            'observed_manifest_sha256': observed_manifest_sha256,
            'required_dataset_version': 1,
            'version_url': 'https://www.kaggle.com/datasets/heliosli/rsna-knee-r6-frozen-fe055406/versions/1'}}
identity_ok = (observed_bundle_id == identity['expected_bundle_id'] and
               observed_manifest_sha256 == expected_manifest_sha256)
identity['status'] = 'VERIFIED' if identity_ok else 'WRONG_DATASET_VERSION'
(WORK / '{tag}_asset_identity.json').write_text(json.dumps(identity, indent=2))
print(json.dumps(identity, indent=2))
if not identity_ok:
    raise RuntimeError(
        'Wrong R6 dataset version: observed bundle=' + str(observed_bundle_id) +
        ', manifest SHA256=' + observed_manifest_sha256 + '. '
        'Attach heliosli/rsna-knee-r6-frozen-fe055406 VERSION 1, not latest/version 2. '
        'Version 2 contains R7 8c7b666e27dc0e8f under the same file names. '
        'Do not bypass the identity guard; see {tag}_asset_identity.json.')''')
    pre += f"\nEXPERIMENT = '{tag}'\nEXPECT_FLIP = {flip!r}\nEXPECT_224 = {use224!r}\n"
    pre += r'''
if not EXPECT_FLIP and r7_manifest.get('flip_tta'):
    raise RuntimeError('The frozen bundle enables TTA by itself; plain ablation requires the original no-TTA bundle')
if EXPECT_224 and not (raptor / 'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt').is_file():
    raise RuntimeError('Attach the publication dataset version containing the exact 224 checkpoint too')
preflight.update(experiment=EXPERIMENT, expect_flip=EXPECT_FLIP, expect_224=EXPECT_224,
                 model_bundle='R6 fe055406fad6a12f')
(WORK / (EXPERIMENT + '_preflight.json')).write_text(json.dumps(preflight, indent=2))
'''
    cells = [cell(intro, True), cell(pre)]
    index_sources = {}
    for i in range(2, 9):
        if not use224 and i in (4, 5):
            continue
        text = s(i)
        if i == 6 and not flip:
            assert text.count('FLIP_TTA = True') == 1
            text = text.replace('FLIP_TTA = True', 'FLIP_TTA = False')
        if i == 8 and not use224:
            text = old_s(7).replace('public Raptor model', 'public Raptor model').replace(
                '# Raptor trained on every training study and was selected on the 58 gold, so no clean labelled set exists to fit them.',
                '# Gold was reused for upstream development/selection; it is not independent validation.')
            text += "\nblend['weights']['a224'] = 0.0\nblend['raptor224_seconds'] = 0.0\nblend['raptor224_failed'] = 0\n(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=2))\n"
        cells.append(cell(text))
        index_sources[i] = text
    audit = r'''
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
work = Path('/kaggle/working')
blend_receipt = json.loads((work / 'blend_receipt.json').read_text())
model_receipt = json.loads((work / 'submission_receipt.json').read_text())
workers = model_receipt.get('workers', [])
actual_flip = [bool(worker.get('flip_tta')) for worker in workers]
if not workers or any(value != EXPECT_FLIP for value in actual_flip):
    raise RuntimeError('R6 worker mirror configuration differs from the selected experiment')
if model_receipt.get('bundle_id') != 'fe055406fad6a12f':
    raise RuntimeError('Wrong R6 bundle in the scored receipt')
frame = pd.read_csv(work / 'submission.csv', dtype={'StudyInstanceUID': str})
ids = pd.read_csv(DATA / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
columns = list(pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
if list(frame.columns) != columns or frame.StudyInstanceUID.tolist() != ids:
    raise RuntimeError('Submission schema or UID order mismatch')
values = frame.iloc[:, 1:].to_numpy(dtype=float)
if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
    raise RuntimeError('Invalid submission values')
failures = {name: blend_receipt.get(name, 0) for name in
            ('raptor384_failed', 'raptor224_failed', 'raptor_failed', 'ours_failed', 'both_failed')}
audit = {'experiment': EXPERIMENT,
         'status': 'COMPLETE' if not any(failures.values()) and model_receipt.get('scored') == len(ids) else 'FALLBACKS_PRESENT',
         'rows_visible_to_this_run': len(ids), 'model_bundle': model_receipt['bundle_id'],
         'expected_flip_tta': EXPECT_FLIP, 'workers_flip_tta': actual_flip,
         'uses_raptor224': EXPECT_224, 'weights': blend_receipt['weights'], 'failures': failures,
         'total_seconds': blend_receipt.get('total_seconds'),
         'submission_sha256': hashlib.sha256((work / 'submission.csv').read_bytes()).hexdigest(),
         'note': 'Saved visible rows and runtime do not establish hidden-test performance.'}
(work / (EXPERIMENT + '_run_audit.json')).write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
'''
    cells.append(cell(audit))
    notebook = copy.deepcopy(ref)
    notebook['cells'] = cells
    notebook['metadata'].pop('papermill', None)
    kaggle = notebook['metadata'].setdefault('kaggle', {})
    kaggle.update(isInternetEnabled=False, isGpuEnabled=True, accelerator='nvidiaTeslaT4')
    # Verified from Kaggle's version-1 manifest download storage path:
    # /kagglesdsdata/datasets/12524886/20611930/manifest.json
    # Do not inherit a mutable/latest dataset reference from a future source edit.
    kaggle['dataSources'] = [
        {'sourceType': 'competition', 'sourceId': 154281},
        {'sourceType': 'datasetVersion', 'sourceId': 20297367},
        {'sourceType': 'datasetVersion', 'sourceId': 20611930},
        {'sourceType': 'datasetVersion', 'sourceId': 20440357},
    ]
    if tag == '29A':
        assert all(index_sources[i] == s(i) for i in range(2, 9)), 'Reference scoring code drift'
    elif tag == '29B':
        assert all(index_sources[i] == s(i) for i in (2, 3, 6, 7))
    else:
        assert all(index_sources[i] == s(i) for i in (2, 3, 4, 5, 7, 8))
    path = ROOT / f'{tag}_{suffix}.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    pinned_path = ROOT / f'{tag}_{suffix}_V1Pinned.ipynb'
    pinned_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    created.append({'file': path.name, 'flip': flip, 'use224': use224,
                    'all_code_compile': True, 'source_cells_checked': True})

# Verify mathematical equivalence of B to the reference blend at zero 224 weight,
# including either/both Raptor failure and missing R6 prediction routes.
def reference_mix(a, b, c, fa=False, fb=False, fc=False, share=0.35):
    half = b if fa else a if fb else (1-share)*a + share*b
    value = .5*half + .5*c
    if fa and fb: value = c
    if fc: value = half
    if fa and fb and fc: value = .5
    return value
for a, b, c in [(0.1,0.8,0.4),(0.9,0.2,0.6),(0.5,0.5,0.5)]:
    assert abs(reference_mix(a,b,c,share=0) - (.5*a+.5*c)) < 1e-12
    assert reference_mix(a,b,c,fc=True,share=0) == a
    assert reference_mix(a,b,c,fa=True,fb=True,fc=True) == .5
(ROOT / '29_probe_manifest.json').write_text(json.dumps({
    'reference_file': str(SOURCE), 'reference_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'reference_scoring_hashes': core_hashes, 'probes': created}, indent=2), encoding='utf-8')
print(json.dumps(created, indent=2))
