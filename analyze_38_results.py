"""Audit user-provided S6 diagnostic output, without extracting/executing its code."""
import hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
archive=ROOT/'results 38.zip'
with zipfile.ZipFile(archive) as z:
    prefix='rsna_38_s6/'
    report=json.loads(z.read(prefix+'38_diagnostic_report.json'))
    def frame(name):
        return pd.read_csv(io.BytesIO(z.read(prefix+name)),dtype={'StudyInstanceUID':str})
    gold=frame('38_s6_gold_probabilities.csv').set_index('StudyInstanceUID')
    metrics=frame('38_per_target_metrics.csv')
    visible=frame('38_s6_visible_probabilities.csv')
    assert report['status']=='COMPLETE'
    assert report['inputs']['model']['sha256']=='0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'
    assert report['packaged_component_sha256']==hashlib.sha256(z.read(prefix+'s6_component_b.py')).hexdigest()
    assert len(visible)==3 and visible.StudyInstanceUID.is_unique
    for stage,count in [('visible',3),('gold',58)]:
        info=report[stage]
        assert not info['failed_after_retry'] and info['retry'] is None
        assert sum(r['scored'] for r in info['receipts'])==count
        assert all(r['timed_out'] is False for r in info['receipts'])
        member=prefix+'worker_outputs/'+stage+'/'+stage+'_probs.csv'
        assert hashlib.sha256(z.read(member)).hexdigest()==info['probs_sha256']
        assert hashlib.sha256(z.read(prefix+'worker_outputs/'+stage+'/'+stage+'_info.json')).hexdigest()==info['info_sha256']
        for i in range(len(info['parts'])):
            r=json.loads(z.read(prefix+f'worker_outputs/{stage}/part{i}/receipt.json'))
            assert r['status']=='PASS' and not r.get('fallback')
            assert r['models']==['E_prior_s42_fall.pt']
            assert r['anchor_weights_sha256']=='5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e'
    for name in z.namelist():
        if name.endswith('log.txt'):
            log=z.read(name).decode(errors='replace')
            assert 'WORKER_DONE' in log
            assert 'OutOfMemoryError' not in log and 'Traceback (most recent call last)' not in log

def old(archive,name):
    with zipfile.ZipFile(ROOT/archive) as z:
        return pd.read_csv(io.BytesIO(z.read(name)),dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
truth=old('results 30.zip','30_gold_truth.csv')
labels=list(truth.columns)
assert gold.index.is_unique and set(gold.index)==set(truth.index) and len(gold)==58
y=truth.to_numpy(int)
s6=gold.loc[truth.index,labels].to_numpy(float)
assert np.isfinite(s6).all() and (s6>=0).all() and (s6<=1).all()
proxy=old('results 30.zip','30_gold_blend_r6_tta_current.csv').loc[truth.index,labels].to_numpy(float)
def auc(p):return np.array([roc_auc_score(y[:,j],p[:,j]) for j in range(12)])
assert abs(auc(s6).mean()-report['macro_auc_descriptive_only']['s6'])<1e-12
assert abs(auc(.9*proxy+.1*s6).mean()-report['macro_auc_descriptive_only']['proxy29B_90_s6_10'])<1e-12
metrics['fixed_blend_delta']=metrics.proxy29B_90_s6_10_auc-metrics.proxy29B_auc
counts={'improved':int((metrics.fixed_blend_delta>1e-12).sum()),
        'worsened':int((metrics.fixed_blend_delta< -1e-12).sum()),
        'unchanged':int((metrics.fixed_blend_delta.abs()<=1e-12).sum())}
summary={'status':'VERIFIED_COMPLETE','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
         'gold_scored':58,'visible_scored':3,'raptor_fallbacks':0,
         'total_seconds':report['elapsed_seconds'],'gold_inference_seconds':report['gold']['seconds'],
         'macro_auc_descriptive_only':report['macro_auc_descriptive_only'],
         'fixed_blend_delta':float(metrics.fixed_blend_delta.mean()),'target_change_counts':counts,
         'paired_macro_auc_delta_ci95':report['paired_macro_auc_delta_ci95'],
         'limitations':report['limitations'],
         'decision':'Worth one preregistered 10% S6 + 90% exact33 Public probe; not proof of a gain. Do not optimize weights on gold.',
         'new_submission_notebook_created':False}
(ROOT/'38_Results_Audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
table=metrics[['target','r6_auc','s6_auc','proxy29B_auc','proxy29B_90_s6_10_auc','fixed_blend_delta','s6_r6_spearman']]
print(json.dumps(summary,indent=2))
print(table.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
lines=['# Notebook38 results — S6 complementarity audit','',
       'Verified output: all 58 gold and 3 visible studies scored; no Raptor fallback, retry or worker timeout. Model and anatomy identities match the pinned release.',
       '',f"Total diagnostic: {report['elapsed_seconds']:.2f}s; gold inference: {report['gold']['seconds']:.1f}s. This is not a hidden-test runtime guarantee.",'',
       '## Descriptive gold results','',
       '| Model | Macro AUC |','|---|---:|']
lines.extend(f'| {name} | {value:.6f} |' for name,value in report['macro_auc_descriptive_only'].items())
lines+=['',f"Fixed 90% proxy29B + 10% S6 changes macro AUC by {summary['fixed_blend_delta']:+.6f}. Paired bootstrap 95% interval: [-0.000933, +0.003619], includes zero. Gold is reused, so these intervals do not remove selection bias.",'',
        '## Per-target fixed-blend changes','', '| Target | Delta AUC | S6–R6 Spearman |','|---|---:|---:|']
lines.extend(f"| {r['target']} | {r['fixed_blend_delta']:+.6f} | {r['s6_r6_spearman']:.3f} |" for r in metrics.to_dict('records'))
lines+=['','## Decision','',
        'S6 is weaker than R6 alone on this reused set, but the fixed blend has a small positive exploratory signal. A single 90% exact-anchor33 + 10% S6 probability-blend probe is justified as a controlled Public experiment, not an expected breakthrough. Keep every internal 33 target weight unchanged and protect 33. Do not tune per-target S6 weights on these results.',
        '', 'The reference here is proxy29B, not exact33: Raptor224 gold is unavailable. A Public result may therefore differ or regress. No new submission notebook has been created by this analysis. All new inference must run on actual hidden test rows; never reuse the three visible predictions as a hidden submission.',
        '', 'The slab_prior anchor method occurred in five gold studies. This is an internal anatomy fallback, not a Raptor model fallback. Do not describe every study as having a successful learned anatomy localization.']
(ROOT/'38_Results_and_Next_Probe.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
