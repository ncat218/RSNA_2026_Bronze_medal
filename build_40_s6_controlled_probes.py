"""Audit results39 and package two one-factor submission probes; no inference."""
import ast, copy, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
base_path=ROOT/'39_Anchor0958_Plus_S6_Fixed10_Rank_Submit.ipynb'
base=json.loads(base_path.read_text(encoding='utf-8'))
base_source=''.join(base['cells'][1]['source'])
manifest=json.loads((ROOT/'39_Build_Manifest.json').read_text())
assert hashlib.sha256(base_path.read_bytes()).hexdigest()==manifest['notebook_sha256']
with zipfile.ZipFile(ROOT/'results 39.zip') as z:
    audit=json.loads(z.read('rsna_39/39_run_audit.json'))
    assert audit['status']=='COMPLETE' and not audit['warnings'] and not audit['s6_failed_uids']
    assert hashlib.sha256(z.read('submission.csv')).hexdigest()==audit['submission_sha256']
    assert hashlib.sha256(z.read('rsna_39/anchor33.py')).hexdigest()==manifest['anchor_adapter_sha256']
    assert hashlib.sha256(z.read('rsna_39/s6_component.py')).hexdigest()==manifest['s6_component_sha256']
    for name in ('s6','anchor33'):
        assert hashlib.sha256(z.read('rsna_39/'+name+'.py')).hexdigest()==audit[name]['script_sha256']
    def df(name):return pd.read_csv(io.BytesIO(z.read(name)),dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
    anchor,s6,final=df('39_anchor33.csv'),df('rsna_39/39_s6_probabilities.csv'),df('submission.csv')
    assert anchor.index.is_unique and set(anchor.index)==set(s6.index)==set(final.index)
    s6=s6.loc[anchor.index,anchor.columns]
    expected=.9*anchor+.1*s6.rank(method='average',pct=True)
    assert np.max(abs(final.loc[anchor.index,anchor.columns].to_numpy()-expected.to_numpy()))<1e-7
    receipt=json.loads(z.read('rsna_39/39_s6_receipt.json'))
    assert receipt['status']=='COMPLETE' and not receipt['failed']
    assert receipt['inputs']['model']['sha256']=='0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'

summary={'experiment':'39_output_audit','status':'VERIFIED_COMPLETE',
         'zip_sha256':hashlib.sha256((ROOT/'results 39.zip').read_bytes()).hexdigest(),
         'user_reported_public_score':.958,'user_reported_public_rank_before':1062,'user_reported_public_rank_after':404,
         'public_full_precision_unavailable':True,'saved_output_rows':len(final),
         'saved_run_seconds':audit['total_seconds'],'saved_run_warnings':audit['warnings'],
         's6_vs38_visible_max_abs_delta':audit['visible_s6_vs38_max_abs_delta'],
         'fusion_recomputed_matches':True,
         'interpretation':'Saved ZIP contains three visible rows, not hidden Public predictions. Rank movement may reflect unshown score digits, ties or leaderboard updates; no Private generalization conclusion.',
         'next_probes':{'40A':'80% exact33 + 20% rank(S6); one dose change from39',
                        '40B':'90% exact33 + 10% raw probability(S6); one representation change from39'}}
(ROOT/'39_Results_Audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

configs=[('40A',.2,'rank','40A_Anchor0958_Plus_S6_Fixed20_Rank_Submit.ipynb'),
         ('40B',.1,'probability','40B_Anchor0958_Plus_S6_Fixed10_Probability_Submit.ipynb')]
outputs={}
for tag,weight,representation,filename in configs:
    # All model subprocess sources and their checkpoints remain byte-identical to39.
    # Keep subprocess artifact paths rsna_39 unchanged: these are literal original
    #39 workers whose bytes/identity are intentionally preserved, not new model code.
    source=base_source
    old="'experiment':'39_anchor33_90_S6rank_10'"
    assert old in source
    source=source.replace(old,f"'experiment':'{tag}_anchor33_S6_{representation}_{weight}'")
    source=source.replace("'fusion':'0.90 * unchanged33_rank_blend + 0.10 * rank(S6)'",
                          "'fusion':"+repr(f'{1-weight:.2f} * unchanged33_rank_blend + {weight:.2f} * '+('rank(S6)' if representation=='rank' else 'probability(S6)')))
    source=source.replace("'note':'Different from the probability blend measured on reused gold; improvement unproven.'",
                          "'note':"+repr('One-factor probe versus39; no fitted weights or assumed Private gain.'))
    old_line='out[ok]=.90*a[ok]+.10*ranks'
    assert source.count(old_line)==1
    source=source.replace(old_line,f'out[ok]={1-weight:.2f}*a[ok]+{weight:.2f}*ranks')
    if representation=='probability':
        old_rank="ranks=pd.DataFrame(b[ok]).rank(method='average',pct=True).to_numpy(float)"
        assert source.count(old_rank)==1
        source=source.replace(old_rank,'ranks=b[ok]  # hypothesis: preserve S6 confidence; calibration mismatch is a risk')
        source=source.replace('# Preserve anchor\'s existing rank blend; rank S6 over its own scored rows only.',
                              '# Preserve anchor rank blend; intentionally use raw S6 confidence for this probe only.')
    # Output directories retain rsna_39 because all inherited subprocesses use this
    #fixed path. Clearly distinguish experiment in the audit and notebook name.
    source=source.replace("print('39 COMPLETE'",f"print('{tag} COMPLETE'")
    note=f'''# {tag} — controlled S6 {'weight' if representation=='rank' else 'confidence'} submission probe

User-confirmed notebook39 Public **0.958**, reported rank404; Private performance is unknown. This is a new unscored probe, not a promised improvement.

Final formula: **{1-weight:.0%} unchanged anchor33 + {weight:.0%} {'S6 percentile ranks' if representation=='rank' else 'raw S6 probabilities'}**. {'Only S6 weight changes from39 (10% → 20%).' if representation=='rank' else 'Only S6 representation changes from39 (rank → probability), retaining 10%. This deliberately mixes confidence with the rank-scale anchor, so calibration can harm the result.'} Every internal33 per-target weight, model, preprocessing and stage source is unchanged. No gold tuning or training.

Attach exactly the **same five inputs/versions as successful39**. No Output39 required. T4x2, Internet Off, Save & Run All, then submit if the audit is COMPLETE and warnings are understood. All stages rerun on actual test data; visible predictions are not reused for hidden scoring.

Inherited artifact paths remain `rsna_39/39_run_audit.json`, `39_anchor33.csv`, `rsna_39/39_s6_probabilities.csv` to preserve the tested model-stage code. Check audit `experiment` starts with **{tag}**. This is intentional. Use separate Kaggle notebooks for40A and40B; each creates its own `submission.csv`. Eight-hour timeout is not an ETA. Keep33 and39 protected.

These two experiments are independent. Run either order; if you submit both, compare them at the same time using full-precision scores if available. Do not infer Private generalization from Public movement alone. Send both scores and Output ZIPs. No GPU/hidden run has yet been verified for these new files.

Credits and third-party licence/provenance conditions are inherited from39 (Nartaa, Heliosli and the original33 authors). Nothing here changes model permissions or resolves eligibility questions.
'''
    nb=copy.deepcopy(base)
    nb['cells'][0]['source']=note.splitlines(keepends=True)
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    nb['cells'][0]['id']=tag+'-intro'
    nb['cells'][1]['id']=tag+'-controller'
    compile(source,'<'+tag+'>','exec')
    path=ROOT/filename
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    outputs[tag]={'file':filename,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'weight':weight,'representation':representation,'gpu_verified':False}
(ROOT/'40_Build_Manifest.json').write_text(json.dumps(outputs,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print(json.dumps(outputs,indent=2))
