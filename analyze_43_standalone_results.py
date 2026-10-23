"""Verify the two user-supplied S6 outputs. Do not execute archived code."""
import ast, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
build=json.loads((ROOT/'43_Standalone_Build_Manifest.json').read_text())
predictions={}
summary={'date':'2026-10-21','status':'VERIFIED_COMPLETE',
         'public_scores_user_screenshot':{'43A':.950,'43B':.950},
         'public_full_precision_available':False,'runs':{}}
submissions={}
raw_bytes={}
for tag in ('43A','43B'):
    path=ROOT/('results '+tag+'.zip')
    prefix='rsna_'+tag+'/'
    with zipfile.ZipFile(path) as z:
        audit=json.loads(z.read(prefix+tag+'_run_audit.json'))
        assert audit['experiment']==tag and audit['status']=='COMPLETE'
        assert audit['standalone_S6'] and not audit['uses_anchor33'] and not audit['uses_Raptor'] and not audit['uses_R6']
        assert not audit['warnings']
        assert hashlib.sha256(z.read('submission.csv')).hexdigest()==audit['submission_sha256']
        component=z.read(prefix+'s6_component.py')
        stage=z.read(prefix+'s6.py')
        assert hashlib.sha256(component).hexdigest()==build[tag]['component_sha256']==audit['component_sha256']
        assert hashlib.sha256(stage).hexdigest()==build[tag]['stage_sha256']==audit['s6']['script_sha256']
        receipt=audit['s6_receipt']
        assert receipt['status']=='COMPLETE' and not receipt['failed'] and not receipt['model_fallback_to_raptor']
        assert receipt['inputs']['model']['sha256']=='0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'
        run=receipt['run']
        assert not run['failed_after_retry'] and run['retry'] is None
        probs=prefix+'s6_outputs/B/B_probs.csv'
        assert hashlib.sha256(z.read(probs)).hexdigest()==run['probs_sha256']
        assert hashlib.sha256(z.read(prefix+'s6_outputs/B/B_info.json')).hexdigest()==run['info_sha256']
        raw=pd.read_csv(io.BytesIO(z.read(prefix+'39_s6_probabilities.csv')),dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
        final=pd.read_csv(io.BytesIO(z.read('submission.csv')),dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
        assert raw.index.is_unique and final.index.is_unique and len(raw)==audit['rows']==3
        assert raw.index.tolist()==final.index.tolist()
        assert np.isfinite(raw.values).all() and ((raw.values>=0)&(raw.values<=1)).all()
        np.testing.assert_allclose(final.to_numpy(),raw.rank(method='average',pct=True).to_numpy(),atol=1e-15,rtol=0)
        workers=[]
        assignments=[]
        for i in range(len(run['parts'])):
            p=prefix+f's6_outputs/B/part{i}/'
            r=json.loads(z.read(p+'receipt.json'))
            assigned=json.loads(z.read(p+'ids.json'))
            assignments.extend(assigned)
            assert r['status']=='PASS' and r['timed_out'] is False
            assert r['studies']==r['scored']==len(assigned) and not r.get('fallback')
            assert r['models']==['E_prior_s42_fall.pt']
            if tag=='43B':
                assert r['crop_tta']=={'offsets':[[8,8],[24,24]],'crop':288,'cache_size':320,
                                       'combine':'equal study-logit mean','passes':2}
            else:assert 'crop_tta' not in r
            log=z.read(p+'log.txt').decode(errors='replace')
            assert 'WORKER_DONE' in log and 'Traceback (most recent call last)' not in log
            workers.append(r)
        assert len(assignments)==len(set(assignments))==len(raw) and set(assignments)==set(raw.index)
        if tag=='43B':
            patch=json.loads(z.read(prefix+'s6_outputs/42_worker_patch_audit.json'))
            assert patch['patched_sha256']=='7da30ee45aa116e3f246fc5c58ab0d25c08a03d62a386618d75a220c7931c446'
        summary['runs'][tag]={
            'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'saved_visible_rows':len(raw),'total_seconds':audit['total_seconds'],
            's6_stage_seconds':audit['s6']['seconds'],'inference_group_seconds':run['seconds'],
            'summed_worker_gpu_seconds':round(sum(r['stats']['gpu_s'] for r in workers),2),
            'window_counts':{u:int(row[1]) for r in workers for row in r.get('per_study',[]) for u in [row[0]]},
            'scored':sum(r['scored'] for r in workers),'model_fallbacks':0,
            'source_identity_verified':True,'csv_reproduced_from_probabilities':True,
            'expected_crop_recipe_verified':True,'visible_vs38_max_abs_probability_delta':audit['visible_vs38_max_abs_probability_delta'],
            'anatomy_methods':{method:sum(r.get('anchor_methods',{}).get(method,0) for r in workers)
                               for method in set().union(*(set(r.get('anchor_methods',{})) for r in workers))}}
        predictions[tag]=raw
        submissions[tag]=final
        raw_bytes[tag]=z.read('submission.csv')

a=predictions['43A']
b=predictions['43B'].loc[a.index,a.columns]
assert summary['runs']['43A']['window_counts']==summary['runs']['43B']['window_counts']
delta=abs(b-a)
rank_a=a.rank(method='average')
rank_b=b.rank(method='average')
per_target=[]
for target in a.columns:
    per_target.append({'target':target,'visible_max_abs_probability_delta':float(delta[target].max()),
                       'visible_mean_abs_probability_delta':float(delta[target].mean()),
                       'visible_rank_changed_rows':int((rank_a[target]!=rank_b[target]).sum())})
comparison={'visible_probability_changed_cells':int((delta.values>1e-12).sum()),
            'visible_probability_cells':int(delta.size),'visible_max_abs_probability_delta':float(delta.values.max()),
            'visible_mean_abs_probability_delta':float(delta.values.mean()),
            'visible_rank_changed_cells':int((rank_a.values!=rank_b.values).sum()),
            'visible_submission_byte_identical':raw_bytes['43A']==raw_bytes['43B'],
            'visible_submission_sha256':hashlib.sha256(raw_bytes['43A']).hexdigest(),
            'same_window_counts':True}
summary['comparison']=comparison
summary['interpretation']=[
    'Both recipes genuinely ran with identical checkpoint and window counts; this is not an accidental duplicate baseline.',
    'Crop TTA changes visible probabilities but does not change ordering on any of the twelve targets across the three visible studies.',
    'Identical saved three-row rank CSVs do NOT imply identical hidden Public predictions.',
    'Both Public scores round to0.950; no demonstrated benefit at displayed precision, not proof of exact equality.',
    'The saved timings are startup/decode dominated three-study observations, not hidden runtime measurements.',
    'No test labels or per-study hidden predictions are in the ZIPs; no hidden error analysis or Private generalization claim is possible.',
    'Stop further crop-offset probing. Keep successful0.958 ensembles as final candidates; do not promote standalone0.950 from recency.']
(ROOT/'43_Standalone_Results_Audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# S6 standalone crop-TTA result audit — 2026-10-21','',
       '## Outcome','',
       'User screenshots confirm Public43A=0.950 and43B=0.950. This audit checks saved visible outputs, not hidden-run logs.',
       '', '| Check |43A centre-only|43B two-crop|','|---|---:|---:|',
       '|Checkpoint|same pinned S6|same pinned S6|',
       '|Crop offsets|(16,16)|(8,8)+(24,24)|',
       '|Saved studies scored|3/3|3/3|',
       '|Model fallback|0|0|',
       f"|Saved total time|{summary['runs']['43A']['total_seconds']:.2f}s|{summary['runs']['43B']['total_seconds']:.2f}s|",
       f"|Sum worker GPU timings|{summary['runs']['43A']['summed_worker_gpu_seconds']:.2f}s|{summary['runs']['43B']['summed_worker_gpu_seconds']:.2f}s|",
       '', '## Important distinction','',
       f"TTA changed {comparison['visible_probability_changed_cells']}/{comparison['visible_probability_cells']} visible probabilities. Maximum absolute change={comparison['visible_max_abs_probability_delta']:.6f}; mean={comparison['visible_mean_abs_probability_delta']:.6f}.",
       '', '**Nevertheless, all12 target orderings on these three studies stayed unchanged, so the two final rank-based CSVs are byte-identical.** This does not mean the worker patch failed; the source/receipt records verify the different crop forwards. It also does not prove the hidden submissions were identical: three-study ordering is not the ordering over the hidden test population.',
       '', '## Decision','',
       'No demonstrated Public gain at displayed precision. Stop testing further crop offsets: current evidence does not justify more Public tuning on this narrow hypothesis. Exact score equality and Private behavior remain unknown. Do not swap out successful0.958 final candidates for these standalone controls.',
       '', 'Runtime comparison is inconclusive for hidden deployment.43B has two actual crop forwards but its saved total is slightly smaller; startup, decode and scheduling variation dominate the tiny sample. Do not market it as faster or extrapolate from this to the hidden set.',
       '', 'Archive source hashes, checkpoint/anatomy receipt identities, strict study coverage and final CSV derivations were verified. One visible study uses the built-in slab_prior anatomy method in both conditions; this is not a Raptor prediction fallback.',
       '', 'The earlier Version1 logs established missing anatomy input before inference. The new successful outputs do not support attributing those earlier failures to cropTTA, OOM or the strict missing-study guard.']
(ROOT/'43_Standalone_Results_and_Decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'runs':summary['runs'],'comparison':comparison},indent=2))
print(pd.DataFrame(per_target).to_string(index=False))
