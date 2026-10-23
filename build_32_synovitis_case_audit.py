"""Package a CPU-only, privacy-conservative case audit from verified predictions."""
import csv
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
def read(z, name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))
with zipfile.ZipFile(ROOT / 'results 30.zip') as z:
    truth = read(z, '30_gold_truth.csv')
    r6 = {r['StudyInstanceUID']: r for r in read(z, '30_r6_tta_gold.csv')}
with zipfile.ZipFile(ROOT / 'results 26.zip') as z:
    rp = {r['StudyInstanceUID']: r for r in read(z, '26_raptor_baseline.csv')}
ref = []
for r in truth:
    uid = r['StudyInstanceUID']
    ref.append({'StudyInstanceUID': uid,
                'gold_synovitis': int(float(r['Synovitis'])),
                'gold_effusion': int(float(r['Effusion'])),
                'r6_synovitis': float(r6[uid]['Synovitis']),
                'r6_effusion': float(r6[uid]['Effusion']),
                'raptor_synovitis': float(rp[uid]['Synovitis']),
                'raptor_effusion': float(rp[uid]['Effusion'])})

intro = '''# 32 — Synovitis / Effusion discordant-case audit (CPU)

Diagnostic only: NO training, NO GPU, NO submission. Attach only the RSNA competition. Predictions from experiments 26/30 are embedded; no output ZIP or model input is required. Use Accelerator None, Internet Off, Save & Run All.

This notebook checks the 58 gold identities and prepares the 18 discordant cases (5 Synovitis-only; 13 Effusion-only), plus all-case scores and a blank manual review worksheet. Scores are not calibrated diagnostic thresholds. Ranks use all 58 reference studies and are not hidden-test submission ranks. Gold was reused upstream and is NOT independent validation.

Privacy: raw reports are NOT written to output files. SHOW_REPORTS=False by default. After the saved run, you may set SHOW_REPORTS=True and run the last cell interactively to read reports inside your private notebook. If you do so, do not publish/export the executed notebook with report text; clear its outputs first. No external translator or LLM API is called. Do not treat this worksheet as clinical advice or rewrite labels automatically.

Return the small Output ZIP and, if reviewed locally, your manual annotations without raw report text. A future label-source audit needs separately pinned source tables; this notebook does not reconstruct the unknown Nartaa label blend.
'''
setup = '''from pathlib import Path
import json
import numpy as np
import pandas as pd

SHOW_REPORTS = False
INPUT = Path('/kaggle/input')
OUT = Path('/kaggle/working/rsna32_case_audit')
OUT.mkdir(parents=True, exist_ok=True)
paths = [INPUT / 'rsna-knee-abnormality-detection', INPUT / 'competitions' / 'rsna-knee-abnormality-detection']
roots = [p for p in paths if (p / 'train.csv').is_file()]
if len(roots) != 1:
    raise RuntimeError('Attach exactly one RSNA competition input containing train.csv')
train = pd.read_csv(roots[0] / 'train.csv', dtype={'StudyInstanceUID': str})
if not train.StudyInstanceUID.is_unique:
    raise RuntimeError('Duplicate training StudyInstanceUID')
''' + 'REF = ' + repr(ref) + '\n'
analysis = '''scores = pd.DataFrame(REF).set_index('StudyInstanceUID')
if len(scores) != 58 or not scores.index.is_unique:
    raise RuntimeError('Unexpected embedded reference inventory')
gold = train.loc[train[['Synovitis', 'Effusion']].notna().all(axis=1)].set_index('StudyInstanceUID')
if set(gold.index) != set(scores.index):
    raise RuntimeError('Gold UID set differs from verified experiments 26/30')
gold = gold.loc[scores.index]
if not np.array_equal(gold[['Synovitis', 'Effusion']].values,
                      scores[['gold_synovitis', 'gold_effusion']].values):
    raise RuntimeError('Gold label identity mismatch')
probability_columns = ['r6_synovitis', 'r6_effusion', 'raptor_synovitis', 'raptor_effusion']
p = scores[probability_columns].to_numpy(float)
if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
    raise RuntimeError('Invalid embedded model probabilities')
for col in probability_columns:
    scores[col+'_rank58'] = scores[col].rank(method='average', pct=True)
scores['blend_synovitis_rank58'] = .5*scores.r6_synovitis_rank58 + .5*scores.raptor_synovitis_rank58
scores['blend_effusion_rank58'] = .5*scores.r6_effusion_rank58 + .5*scores.raptor_effusion_rank58
scores['case_group'] = np.select(
    [(scores.gold_synovitis == 1) & (scores.gold_effusion == 0),
     (scores.gold_synovitis == 0) & (scores.gold_effusion == 1),
     (scores.gold_synovitis == 1) & (scores.gold_effusion == 1)],
    ['synovitis_only', 'effusion_only', 'both_positive'], default='both_negative')
counts = scores.case_group.value_counts().to_dict()
expected = {'synovitis_only':5, 'effusion_only':13, 'both_positive':22, 'both_negative':18}
if counts != expected:
    raise RuntimeError('Discordant-case inventory changed')
cases = scores.loc[scores.case_group.isin(['synovitis_only', 'effusion_only'])].copy()
cases = cases.sort_values(['case_group', 'blend_synovitis_rank58'])
scores.to_csv(OUT / '32_all58_scores.csv', index_label='StudyInstanceUID')
cases.to_csv(OUT / '32_discordant18_scores.csv', index_label='StudyInstanceUID')
review = cases[['case_group', 'gold_synovitis', 'gold_effusion']].copy()
for col in ['report_language', 'synovitis_evidence_explicit_implied_absent_unclear',
            'effusion_evidence_explicit_implied_absent_unclear',
            'severity_or_uncertainty', 'image_review_done', 'report_gold_agreement',
            'suspected_failure_mechanism', 'reviewer_notes_no_raw_report']:
    review[col] = ''
review.to_csv(OUT / '32_manual_review_template.csv', index_label='StudyInstanceUID')
summary = {'experiment':'32_synovitis_effusion_case_audit', 'status':'COMPLETE',
           'gold_studies':58, 'case_counts':counts, 'discordant_cases':len(cases),
           'training_started':False, 'submission_created':False, 'raw_reports_exported':False,
           'gold_is_independent_validation':False,
           'note':'Manual review required. No new labels, fitted weights, residual-based prediction changes or error diagnoses are generated.'}
(OUT / '32_audit_summary.json').write_text(json.dumps(summary, indent=2))
display(cases.round(5))
print(json.dumps(summary, indent=2))
'''
view = '''# Optional private review. Raw reports remain outside exported CSV/JSON files.
if SHOW_REPORTS:
    if 'Report' not in gold:
        raise RuntimeError('Competition train.csv has no Report column')
    print('PRIVATE REVIEW: clear this cell output before sharing or publishing the notebook.')
    for uid in cases.index:
        print('\\nSTUDY', uid, 'GROUP', cases.loc[uid, 'case_group'])
        print(str(gold.loc[uid, 'Report']))
else:
    print('Raw report display disabled. Review template created; no raw reports exported.')
'''
def cell(text, markdown=False):
    if not markdown:
        compile(text, '<32 cell>', 'exec')
    c = {'cell_type':'markdown' if markdown else 'code', 'metadata':{}, 'source':text.splitlines(keepends=True)}
    if not markdown:
        c.update(execution_count=None, outputs=[])
    return c
nb = {'nbformat':4, 'nbformat_minor':5,
      'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                  'language_info':{'name':'python'},
                  'kaggle':{'isGpuEnabled':False,'isInternetEnabled':False,'language':'python',
                            'sourceType':'notebook','dataSources':[{'sourceType':'competition','sourceId':154281}]}},
      'cells':[cell(intro,True),cell(setup),cell(analysis),cell(view)]}
path = ROOT / '32_Synovitis_Effusion_Discordant_Case_Audit_CPU.ipynb'
assert not path.exists(), 'Do not overwrite an existing notebook'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(path.name)
