"""Build five fixed fusion-only probes from the successful40B controller.

Model inference sources are byte-preserved. The anchor child is an exact original
prefix followed ONLY by a JSON coverage export. No checkpoint/augmentation changes.
"""
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / '40B_Anchor0958_Plus_S6_Fixed10_Probability_Submit.ipynb'
base = json.loads(BASE.read_text(encoding='utf-8'))
parent = ''.join(base['cells'][1]['source'])
tree = ast.parse(parent)
def literal(name):
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))
original_anchor = literal('ANCHOR_SCRIPT')
anchor_sha = hashlib.sha256(original_anchor.encode()).hexdigest()
assert anchor_sha == '0993187ed79de621f21407272b63e6f2b7491baec968709e9b7ac30db55f3969'
helper = (ROOT / 'rsna45_fusion.py').read_text(encoding='utf-8')
compile(helper, '<45 fusion>', 'exec')
sidecar = '''
# 45 adapter: coverage evidence ONLY. All preceding inference/fusion is unchanged33.
coverage45 = {
    'schema': '45_coverage_v1',
    'original_anchor_script_sha256': __SHA__,
    'uids': test_ids, 'labels': labels,
    'failed384': r384_fail.tolist(), 'failed224': r224_fail.tolist(),
    'failed6': our_fail.tolist(), 'a224': A224,
    'default_raptor_weight': W_RAPTOR, 'raptor_per_class': W_RAPTOR_C,
}
(WORK/'45_model_coverage.json').write_text(json.dumps(coverage45,indent=2,allow_nan=False))
'''.replace('__SHA__', repr(anchor_sha))
adapter = original_anchor + '\n' + sidecar
compile(adapter, '<45 anchor coverage adapter>', 'exec')

SAVE_REPORT = '''def save_report():
    REPORT['total_seconds']=round(time.time()-START,2)
    safe=diagnostic_json_safe(REPORT)
    payload=json.dumps(safe,indent=2,allow_nan=False)
    for filename in ('39_run_audit.json','45_run_audit.json'):
        (OUT/filename).write_text(payload)
'''

SPECS = {
    '45A': ('AnchorPercentile', 'Normalize the compressed anchor rank blend before adding S6 confidence',
            '.90 * rank(anchor33) + .10 * probability(S6)', 1),
    '45B': ('ProbabilityPool', 'Test whether component confidence adds information discarded by ranks',
            '.90 * [w_j*(.65*p384+.35*p224)+(1-w_j)*p6] + .10*pS6', 3),
    '45C': ('LogitPool', 'Test fixed geometric-odds pooling; calibration mismatch is a material risk',
            'sigmoid(.90*[w_j*(.65*logit(p384)+.35*logit(p224))+(1-w_j)*logit(p6)]+.10*logit(pS6)); epsilon=1e-5', 4),
    '45D': ('UniformFamily', 'Remove per-finding overrides to test less class-specific adaptation',
            '.90 * [.50*(.65*rank(p384)+.35*rank(p224))+.50*rank(p6)] + .10*pS6', 2),
    '45E': ('NoRaptor224', 'Test whether the lower-resolution Raptor component dilutes the current ensemble',
            '.90 * [w_j*rank(p384)+(1-w_j)*rank(p6)] + .10*pS6', 5),
}
manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'base40B_sha256': hashlib.sha256(BASE.read_bytes()).hexdigest(),
            'original_anchor_script_sha256': anchor_sha,
            'adapter_sha256': hashlib.sha256(adapter.encode()).hexdigest(),
            'fusion_helper_sha256': hashlib.sha256(helper.encode()).hexdigest(),
            'predeclared_before_new_scores': True, 'training': False,
            'additional_model_forwards_per_run': 0,
            'each_notebook_reexecutes_full40B_models': True,
            's6_component_sha256': hashlib.sha256(literal('S6_COMPONENT').encode()).hexdigest(),
            's6_script_sha256': hashlib.sha256(literal('S6_SCRIPT').encode()).hexdigest(),
            'gpu_or_hidden_execution_verified': False, 'experiments': {}}
for tag, (short, hypothesis, formula, priority) in SPECS.items():
    edits = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ANCHOR_SCRIPT' for t in n.targets):
            edits.append((n.lineno-1, n.end_lineno, 'ANCHOR_SCRIPT = ' + repr(adapter) + '\n'))
        if isinstance(n, ast.FunctionDef) and n.name == 'save_report':
            edits.append((n.lineno-1, n.end_lineno, SAVE_REPORT))
    lines = parent.splitlines(keepends=True)
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    source = ''.join(lines)
    source = source.replace("'40B_anchor33_S6_probability_0.1'", repr(tag + '_fixed_fusion_probe'))
    source = source.replace("'fusion':'0.90 * unchanged33_rank_blend + 0.10 * probability(S6)'", "'fusion':" + repr(formula))
    source = source.replace("'note':'One-factor probe versus39; no fitted weights or assumed Private gain.'",
                            "'note':" + repr(hypothesis + '; no independent CV or Private gain claimed.'))
    # Functions execute only after the helper is defined, including error-report paths.
    marker = '\nclear_partial_submission()\ntry:'
    assert source.count(marker) == 1
    injected = '\nFUSION_COMPONENT = ' + repr(helper) + '\nexec(FUSION_COMPONENT,globals())\n'
    injected += "REPORT.update(independent_validation=False,public_selection_overfit_risk=True,predeclared_experiment=" + repr(tag) + ")\n"
    source = source.replace(marker, injected + marker)
    old = '    result=fuse_anchor_s6(anchor_frame,s6,ids,labels,failed)'
    new = '''    baseline=fuse_anchor_s6(anchor_frame,s6,ids,labels,failed)
    coverage_path=WORK/'45_model_coverage.json'
    coverage=json.loads(coverage_path.read_text())
    receipt45=json.loads((WORK/'blend_receipt.json').read_text())
    raw45=[pd.read_csv(WORK/name,dtype={'StudyInstanceUID':str})
           for name in ('raptor_raw.csv','raptor224_raw.csv','ours_sub.csv')]
    result,fusion_audit=final_fusion_probe(__TAG__,anchor_frame,s6,*raw45,ids,labels,failed,
                                          coverage,receipt45,baseline,__SHA__)
    REPORT['fusion_audit']=fusion_audit
    REPORT['coverage_sha256']=hashlib.sha256(coverage_path.read_bytes()).hexdigest()
    REPORT['original_anchor_script_sha256']=__SHA__
    baseline.to_csv(OUT/'45_baseline40B.csv',index=False)
    (OUT/'45_fusion_audit.json').write_text(json.dumps(fusion_audit,indent=2,allow_nan=False))'''
    assert source.count(old) == 1
    source = source.replace(old, new.replace('__TAG__', repr(tag)).replace('__SHA__', repr(anchor_sha)))
    source = source.replace('    if time.time()>DEADLINE:raise RuntimeError',
                            "    if fusion_audit['baseline40B_fallback_rows']:warnings.append('45_INCOMPLETE_COVERAGE_ROWS_RETAIN_EXACT40B')\n    if time.time()>DEADLINE:raise RuntimeError")
    source = source.replace("print('40B COMPLETE',json.dumps(REPORT),flush=True)",
                            "print(" + repr(tag + ' COMPLETE') + ",json.dumps(diagnostic_json_safe(REPORT),allow_nan=False),flush=True)")
    # Capture process handles while the leader lives, so detached children can be reaped.
    source = source.replace('        last=0\n        try:', '        last=0\n        seen_children={}\n        try:')
    poll_marker = '            while p.poll() is None:\n'
    source = source.replace(poll_marker, poll_marker + '''                try:
                    import psutil
                    for child in psutil.Process(p.pid).children(recursive=True):
                        seen_children[(child.pid,child.create_time())]=child
                except (ImportError,ProcessLookupError):pass
                except Exception:pass
''')
    source = source.replace('                descendants=psutil.Process(p.pid).children(recursive=True)',
                            '''                descendants=list(seen_children.values())
                try:
                    for child in psutil.Process(p.pid).children(recursive=True):
                        if child not in descendants:descendants.append(child)
                except psutil.NoSuchProcess:pass''')
    compile(source, '<' + tag + ' controller>', 'exec')
    intro = f'''# {tag} — {short}: fixed fusion-only submission probe

**Creates submission.csv; NOT a verified improvement. Priority {priority}/5.** {hypothesis}.
Formula: `{formula}`. `w_j` is the unchanged33 per-target Raptor share (default .5; Lateral OA .2, PF OA .6, Synovitis .8, Baker's .35). All formulas are frozen before these five scores. Probability/logit pools are calibration-sensitive; none has clean CV. Five files are alternatives, not an obligation to consume five slots.

Anchor33, 39, 40B, 41 and 44A were reported Public0.958; repeated LB/gold selection creates overfit risk. These notebooks neither detect nor eliminate it. No labels/gold/report text, training, new checkpoints, crop changes or extra model forwards relative to40B. EACH notebook still reruns the full40B GPU pipeline, not just CPU fusion of saved CSVs. The successful40B inference sources are byte-preserved; the anchor child only appends coverage JSON. Both R6 postprocessors remain ON (this is not44A). In the formulas, p6 means deployed R6 probabilities AFTER the existing postprocessors, not raw head scores.

Attach the SAME FIVE inputs/versions as successful40B:
1. RSNA competition (test.csv/test_series).
2. Nartaa publication SWA weights **Version1** (both384 and224 checkpoints).
3. Exact R6 frozen **w1/v1** bundle from33/40B — NOT latest/R7.
4. Heliosli OrthoFoundation code/anatomy **Version1**, including anatomy.pt.
5. Nartaa S6 all-data weights **Version5**.

**No prior notebook Output or study cache. GPU T4x2, Internet Off; Save & Run All, then submit that saved version once.** Imported metadata pins versions but does not prove an existing draft has replaced its inputs. Keep39/40B selected until results are reviewed. Official final deadline23:59 UTC22 October2026 =06:59AM23 October Bangkok/Vietnam. Check queue/quota: five hidden reruns may not fit. Recommended order45A,45D,45B,45C,45E.

Incomplete-coverage rows retain the exact40B output; failure caps are not relaxed. Component ranks use their own scored UID populations. Audit reconstructs33 from the raw components, asserts weights/UIDs, rejects nonfinite scored probabilities and exports `rsna_39/45_baseline40B.csv` for comparison. Only undefined diagnostic correlations become JSON null, never predictions. No full-component silent fallback.

Before submission require `rsna_39/45_run_audit.json`: status COMPLETE, experiment `{tag}_fixed_fusion_probe`, inspect warnings and `fusion_audit`. Paths retain `rsna_39` for compatibility; the experiment field identifies the new run. Send full Output ZIP and exact Public score/rank. These saved outputs contain visible studies only; hidden predictions are recomputed by Kaggle, not reused from an uploaded CSV.

Eight-hour global cap is a safety timeout, NOT an ETA or hidden-run guarantee. GPU/hidden execution is not locally verified. No0.96+/PrivateTop10 promise. Credits/licences and external-data provenance remain inherited from33/40B; this experiment does not resolve eligibility.
'''
    nb = copy.deepcopy(base)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    nb['cells'][0].update(id=tag+'-intro', source=intro.splitlines(keepends=True))
    nb['cells'][1].update(id=tag+'-controller', source=source.splitlines(keepends=True))
    filename = tag + '_' + short + '_Submit.ipynb'
    path = ROOT / filename
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    manifest['experiments'][tag] = {'file': filename, 'priority': priority,
                                   'formula': formula, 'hypothesis': hypothesis,
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(ROOT/'45_Build_Manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
