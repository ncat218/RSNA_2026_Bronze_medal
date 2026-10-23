"""Independent CPU-only validation of the final45 fusion pack, not model accuracy."""
import ast
import copy
import hashlib
import io
import contextlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
# Deliberately duplicate constants here rather than import the implementation's values.
CLASS_WEIGHTS = {'Medial OA': .5, 'Lateral OA': .2, 'PF OA': .6,
                 'Synovitis': .8, "Baker's": .35}
EXPECTED33 = '685e6e0e65c3c5ed29b11c0b584887d894cd07f274de0b9f83370465ab7b4fde'


def literal(source, name):
    return ast.literal_eval(next(n.value for n in ast.parse(source).body
                                if isinstance(n, ast.Assign) and any(
                                    isinstance(t, ast.Name) and t.id == name for t in n.targets)))


def frame(values, ids, reorder=False):
    result = pd.DataFrame(values, columns=LABELS)
    result.insert(0, 'StudyInstanceUID', ids)
    return result.iloc[::-1].reset_index(drop=True) if reorder else result


def rank_oracle(values, failed):
    """Columnwise independent rank oracle, including a different population per model."""
    output = np.full_like(values, np.nan, dtype=float)
    for j in range(len(LABELS)):
        valid_indices = np.flatnonzero(~failed)
        ranked = pd.Series(values[valid_indices, j]).rank(method='average', pct=True)
        output[valid_indices, j] = ranked.to_numpy()
    return output


def make_fixture(n, partial=False, constant=False):
    rng = np.random.default_rng(9152 + n)
    ids = [f'fixture_{i:05d}' for i in range(n)]
    arrays = [np.round(rng.uniform(.02, .98, size=(n, len(LABELS))), 2) for _ in range(4)]
    for k, array in enumerate(arrays):
        array[:, 1] = .12 + .1 * k  # Constant columns exercise undefined correlations.
        array[0, 0] = 0
        if n > 1:
            array[1, 0] = 1
        if n > 2:
            array[2, 0] = 1  # Explicit ties and both probability endpoints.
        if constant:
            array[:] = .12 + .1 * k
    masks = [np.zeros(n, dtype=bool) for _ in range(4)]
    if partial:
        assert n >= 100
        for k, positions in enumerate(([0, 11], [1, 11], [2, 11], [3, 11])):
            masks[k][positions] = True
        arrays[0][masks[0]] = .5
        arrays[1][masks[1]] = .5
        arrays[2][masks[2]] = .37  # R6 failed studies are mean-filled, not sentinel .5.
        arrays[3][masks[3]] = np.nan
    f384, f224, f6, fs = masks
    r384, r224, r6 = [rank_oracle(p, f) for p, f in zip(arrays, masks)][:3]
    weights = np.array([CLASS_WEIGHTS.get(label, .5) for label in LABELS])
    a = np.empty_like(arrays[0])
    for i in range(n):
        if f384[i] and f224[i] and f6[i]:
            a[i] = .5
        elif f6[i]:
            a[i] = r224[i] if f384[i] else r384[i] if f224[i] else .65*r384[i]+.35*r224[i]
        elif f384[i] and f224[i]:
            a[i] = r6[i]
        else:
            rap = r224[i] if f384[i] else r384[i] if f224[i] else .65*r384[i]+.35*r224[i]
            a[i] = weights*rap + (1-weights)*r6[i]
    # The real anchor exports float32 before the parent controller reads its CSV.
    a = a.astype(np.float32).astype(float)
    baseline = a.copy()
    baseline[~fs] = .9*a[~fs]+.1*arrays[3][~fs]
    baseline = baseline.astype(np.float32).astype(float)
    cov = {'schema': '45_coverage_v1', 'original_anchor_script_sha256': ANCHOR_SHA,
           'uids': ids, 'labels': LABELS, 'failed384': f384.tolist(),
           'failed224': f224.tolist(), 'failed6': f6.tolist(), 'a224': .35,
           'default_raptor_weight': .5, 'raptor_per_class': dict(CLASS_WEIGHTS)}
    rec = {'raptor384_failed': int(f384.sum()), 'raptor224_failed': int(f224.sum()),
           'ours_failed': int(f6.sum()), 'constant_rho': float('nan')}
    return {'ids': ids, 'arrays': arrays, 'masks': masks, 'ranks': [r384, r224, r6],
            'a': a, 'base': baseline, 'coverage': cov, 'receipt': rec,
            'anchor': frame(a, ids, True), 's6': frame(arrays[3], ids, True),
            'raw384': frame(arrays[0], ids, True), 'raw224': frame(arrays[1], ids),
            'raw6': frame(arrays[2], ids, True),
            'baseline': frame(baseline, ids, True),
            'failed': {u: 'injected decode failure' for u, failed in zip(ids, fs) if failed}}


def args(data):
    return [data['anchor'], data['s6'], data['raw384'], data['raw224'], data['raw6'],
            data['ids'], LABELS, data['failed'], data['coverage'], data['receipt'],
            data['baseline'], ANCHOR_SHA]


def oracle(tag, data):
    p384, p224, p6, ps = data['arrays']
    r384, r224, r6 = data['ranks']
    a = data['a']
    w = np.array([CLASS_WEIGHTS.get(label, .5) for label in LABELS])
    e = ~np.logical_or.reduce(data['masks'])
    if tag == '45A':
        fresh = .9*rank_oracle(a, np.zeros(len(a), dtype=bool))+.1*ps
    elif tag == '45B':
        fresh = .9*(w*(.65*p384+.35*p224)+(1-w)*p6)+.1*ps
    elif tag == '45C':
        logits = []
        for p in (p384, p224, p6, ps):
            q = np.maximum(1e-5, np.minimum(1-1e-5, p))
            logits.append(np.log(q/(1-q)))
        z = .9*(w*(.65*logits[0]+.35*logits[1])+(1-w)*logits[2])+.1*logits[3]
        fresh = np.exp(z)/(1+np.exp(z))
    elif tag == '45D':
        fresh = .9*(.5*(.65*r384+.35*r224)+.5*r6)+.1*ps
    else:
        fresh = .9*(w*r384+(1-w)*r6)+.1*ps
    result = data['base'].copy()
    result[e] = fresh[e]
    return result.astype(np.float32), e


base = json.loads((ROOT/'40B_Anchor0958_Plus_S6_Fixed10_Probability_Submit.ipynb').read_text(encoding='utf-8'))
base_source = ''.join(base['cells'][1]['source'])
base_anchor = literal(base_source, 'ANCHOR_SCRIPT')
ANCHOR_SHA = hashlib.sha256(base_anchor.encode()).hexdigest()
manifest = json.loads((ROOT/'45_Build_Manifest.json').read_text())
helper_source = (ROOT/'rsna45_fusion.py').read_text(encoding='utf-8')
scope = {}
exec(compile(helper_source, '<45 actual fusion helper>', 'exec'), scope)
fuse = scope['final_fusion_probe']
assert hashlib.sha256((ROOT/'33_RSNA_0958_R6_Raptor_Submit.ipynb').read_bytes()).hexdigest() == EXPECTED33
assert ANCHOR_SHA == manifest['original_anchor_script_sha256']
tests = []
for n in (1, 2, 3, 1513):
    data = make_fixture(n, partial=n > 100)
    for tag in manifest['experiments']:
        result, audit = fuse(tag, *args(data))
        expected, eligible = oracle(tag, data)
        assert result.StudyInstanceUID.tolist() == data['ids']
        np.testing.assert_allclose(result[LABELS], expected, rtol=0, atol=1e-7)
        assert audit['candidate_rows'] == int(eligible.sum())
        assert audit['baseline40B_fallback_rows'] == int((~eligible).sum())
        # Fallback is a float32 byte-identical copy of exact40B, not a recomputed imputation.
        assert np.array_equal(result[LABELS].to_numpy()[~eligible], data['base'].astype(np.float32)[~eligible])
        assert np.isfinite(result[LABELS].to_numpy()).all()
tests.append('independent five-formula oracle, sizes1/2/3/1513, ties/endpoints/reorderedIDs and own-population ranks')


def rejects(mutator, message=None):
    data = make_fixture(1513, partial=True)
    mutator(data)
    try:
        fuse('45B', *args(data))
    except (RuntimeError, AssertionError) as exc:
        if message:
            assert message in str(exc), (message, str(exc))
    else:
        raise AssertionError('Invalid fixture accepted: '+repr(mutator))


def set_column(data, key, column, values):
    target = data[key]
    target[column] = values


rejects(lambda d: d['ids'].__setitem__(0, d['ids'][1]), 'canonical')
rejects(lambda d: set_column(d, 'anchor', 'StudyInstanceUID', ['wrong']+d['anchor'].StudyInstanceUID.tolist()[1:]), 'UID coverage')
rejects(lambda d: d.__setitem__('s6', pd.concat([d['s6'], d['s6'].iloc[:1]])), 'UID coverage')
rejects(lambda d: set_column(d, 'raw224', 'ACL', np.nan), 'invalid scored')
rejects(lambda d: set_column(d, 'raw6', 'ACL', 1.01), 'invalid scored')
rejects(lambda d: d['coverage'].__setitem__('uids', list(reversed(d['ids']))), 'UID/order')
rejects(lambda d: d['coverage'].__setitem__('labels', list(reversed(LABELS))), 'UID/order')
rejects(lambda d: d['coverage'].__setitem__('schema', 'wrong'), 'schema')
rejects(lambda d: d['coverage'].__setitem__('original_anchor_script_sha256', 'wrong'), 'source identity')
rejects(lambda d: d['coverage'].__setitem__('failed384', [0]*len(d['ids'])), 'boolean coverage')
rejects(lambda d: d['coverage'].__setitem__('failed224', [False]), 'boolean coverage')
rejects(lambda d: d['receipt'].__setitem__('ours_failed', 999), 'disagreement')
rejects(lambda d: d['coverage'].__setitem__('a224', .5), 'anchor weights')
rejects(lambda d: d['coverage'].__setitem__('raptor_per_class', {}), 'class overrides')
rejects(lambda d: d['failed'].__setitem__('not_an_input_uid', 'error'), 'S6 failure')
rejects(lambda d: d.__setitem__('failed', []), 'S6 failure')
rejects(lambda d: d['s6'].loc.__setitem__((d['s6'].StudyInstanceUID == d['ids'][3], LABELS), .3), 'unexpectedly')
rejects(lambda d: d['raw384'].loc.__setitem__((d['raw384'].StudyInstanceUID == d['ids'][7], LABELS), .5), 'sentinel')
rejects(lambda d: set_column(d, 'baseline', 'ACL', .1), 'exact successful40B')
def corrupt_anchor(data):
    data['anchor'].loc[data['anchor'].StudyInstanceUID == data['ids'][8], 'ACL'] += .01
    a = data['anchor'].set_index('StudyInstanceUID').loc[data['ids'], LABELS].to_numpy().copy()
    b = data['s6'].set_index('StudyInstanceUID').loc[data['ids'], LABELS].to_numpy()
    fs = data['masks'][3]
    a[~fs] = .9*a[~fs]+.1*b[~fs]
    data['baseline'] = frame(a.astype(np.float32), data['ids'])
rejects(corrupt_anchor, 'reconstruct exact33')
def excessive_failures(data):
    data['coverage']['failed384'] = [True]*len(data['ids'])
    data['receipt']['raptor384_failed'] = len(data['ids'])
rejects(excessive_failures, 'failure limit')
tests.append('invalid UID/nonfinite/out-of-range/mask/receipt/source/weight/sentinel/baseline/reconstruction rejection')

sources = {}
for tag, item in manifest['experiments'].items():
    path = ROOT/item['file']
    nb = json.loads(path.read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    assert nb['metadata'] == base['metadata']
    source = ''.join(nb['cells'][1]['source'])
    sources[tag] = source
    assert literal(source, 'S6_COMPONENT') == literal(base_source, 'S6_COMPONENT')
    assert literal(source, 'S6_SCRIPT') == literal(base_source, 'S6_SCRIPT')
    assert literal(source, 'FUSION_COMPONENT') == helper_source
    anchor_adapter = literal(source, 'ANCHOR_SCRIPT')
    assert anchor_adapter.startswith(base_anchor+'\n')
    suffix = anchor_adapter[len(base_anchor):]
    assert '45_model_coverage.json' in suffix
    assert len(ast.parse(suffix).body) == 2  # Coverage assignment and JSON write only.
    for text in (source, anchor_adapter, literal(source, 'S6_COMPONENT'), literal(source, 'S6_SCRIPT')):
        compile(text, '<'+tag+' compile>', 'exec')
tests.append('all five source hashes/compilation, exact S6 literals, exact original anchor prefix, coverage-only suffix, metadata and protected33')

FAKE_STAGE = '''
def run_stage(name,script,max_seconds):
    global DEADLINE
    if name=='anchor33' and FIXTURE_CASE=='stage_failure':
        (WORK/'submission.csv').write_text('PARTIAL SHOULD BE REMOVED')
        raise RuntimeError('injected model stage failure')
    if name=='s6':
        FIXTURE['s6'].to_csv(OUT/'39_s6_probabilities.csv',index=False)
        (OUT/'39_s6_receipt.json').write_text(json.dumps({'status':'COMPLETE','failed':FIXTURE['failed']}))
    else:
        FIXTURE['anchor'].to_csv(WORK/'39_anchor33.csv',index=False)
        for key,name in [('raw384','raptor_raw.csv'),('raw224','raptor224_raw.csv'),('raw6','ours_sub.csv')]:
            FIXTURE[key].to_csv(WORK/name,index=False)
        (WORK/'45_model_coverage.json').write_text(json.dumps(FIXTURE['coverage']))
        (WORK/'blend_receipt.json').write_text(json.dumps(FIXTURE['receipt']))
        if FIXTURE_CASE=='late_error':
            (WORK/'submission.csv').write_text('PARTIAL SHOULD BE REMOVED')
            DEADLINE=time.time()-1
    REPORT[name]={'fixture_only':True,'script_sha256':hashlib.sha256(script.encode()).hexdigest()}
'''

with tempfile.TemporaryDirectory(prefix='rsna45_validation_', dir=ROOT) as temp:
    tmp = Path(temp)
    comp = tmp/'competition'
    comp.mkdir()
    for name in ('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt',
                 'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt'):
        (comp/name).write_bytes(b'CPU fixture, not a checkpoint')
    fixture_module = '''
from pathlib import Path
COMP='competition'
HELIOS_SLUG='anatomy'
DS_SLUG='s6'
LAB=__LABELS__
def find_mount(slug,marker):return Path(__COMP__)
def mount_inventory():return ['CPU fixture only']
def b_fail_cap(n):return max(1,int(n*.03))
def sha256_file(path):
    if 'r384_swa.pt' in path.name:return '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
    if 'r224_swa.pt' in path.name:return '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'
    if path.name=='manifest.json':return '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648'
    return 'b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22'
'''.replace('__LABELS__', repr(LABELS)).replace('__COMP__', repr(str(comp)))
    for tag, source in sources.items():
        for case in ('success', 'constant', 'partial_coverage', 'stage_failure', 'late_error'):
            data = make_fixture(1513 if case == 'partial_coverage' else 12,
                                partial=case == 'partial_coverage', constant=case == 'constant')
            pd.DataFrame({'StudyInstanceUID': data['ids']}).to_csv(comp/'test.csv', index=False)
            work = tmp/(tag+'_'+case)
            work.mkdir()
            tree = ast.parse(source)
            stage = ast.parse(FAKE_STAGE).body[0]
            for n in tree.body:
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'S6_COMPONENT' for t in n.targets):
                    n.value = ast.Constant(fixture_module)
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'WORK' for t in n.targets):
                    n.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(str(work))], keywords=[])
            tree.body = [copy.deepcopy(stage) if isinstance(n, ast.FunctionDef) and n.name == 'run_stage' else n for n in tree.body]
            tree = ast.fix_missing_locations(tree)
            env = {'__name__': '__main__', 'display': lambda x: None,
                   'FIXTURE': data, 'FIXTURE_CASE': case}
            failure = case in ('stage_failure', 'late_error')
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    exec(compile(tree, '<'+tag+' full parent '+case+'>', 'exec'), env)
                except RuntimeError as exc:
                    assert failure, (tag, case, str(exc))
                    assert ('injected model stage failure' in str(exc)) if case == 'stage_failure' else ('Deadline exceeded' in str(exc))
                else:
                    assert not failure, (tag, case)
            outdir = work/'rsna_39'
            payload = (outdir/'45_run_audit.json').read_text()
            audit = json.loads(payload)
            assert json.loads((outdir/'39_run_audit.json').read_text()) == audit
            assert 'NaN' not in payload and 'Infinity' not in payload
            assert audit['status'] == ('FAILED' if failure else 'COMPLETE')
            assert audit['experiment'] == tag+'_fixed_fusion_probe'
            assert (work/'submission.csv').exists() == (not failure)
            if not failure:
                final = pd.read_csv(work/'submission.csv')
                expected, eligible = oracle(tag, data)
                assert final.StudyInstanceUID.tolist() == data['ids']
                np.testing.assert_allclose(final[LABELS], expected, rtol=0, atol=1e-7)
                assert audit['anchor33_blend_receipt']['constant_rho'] is None
                assert audit['fusion_audit']['baseline40B_fallback_rows'] == int((~eligible).sum())
                assert hashlib.sha256((work/'submission.csv').read_bytes()).hexdigest() == audit['submission_sha256']
                if case == 'partial_coverage':
                    assert '45_INCOMPLETE_COVERAGE_ROWS_RETAIN_EXACT40B' in audit['warnings']
tests.append('complete clean-parent execution for every45A-E: success/constant/1513-row partial coverage/model failure/late deadline error')
tests.append('diagnostic NaN becomes JSON null, final CSV stays finite, both audits agree, late/stage failures clear partial submission')

result = {'status': 'PASS', 'tests': tests, 'formula_tags': list(sources),
          'largest_fixture_rows': 1513, 'full_parent_fixture_cases': 25,
          'notebook_sha256': {tag: item['sha256'] for tag, item in manifest['experiments'].items()},
          'model_and_mount_stages_mocked': True, 'gpu_execution_tested': False,
          'dicom_decoding_tested': False, 'hidden_submission_tested': False,
          'accuracy_or_generalization_tested': False}
(ROOT/'45_Local_Validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
