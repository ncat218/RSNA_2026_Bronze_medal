"""Two independent R6 postprocessor ablations against successful40B."""
import ast, copy, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
base_path=ROOT/'40B_Anchor0958_Plus_S6_Fixed10_Probability_Submit.ipynb'
base=json.loads(base_path.read_text(encoding='utf-8'))
parent=''.join(base['cells'][1]['source'])
def node(text,name):
    return next(n for n in ast.parse(text).body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
anchor_node=node(parent,'ANCHOR_SCRIPT')
original=ast.literal_eval(anchor_node.value)
guard="deadline = T0 + DEADLINE_HOURS * 3600"
assert original.count(guard)==1
manifest={}
for tag,flag,target in [('44A','--fluid-stacker','Effusion'),('44B','--state-stacker',"Baker's")]:
    # Optional postprocessing switches already supported by the exact deployed worker.
    # Do not alter its weights, input contract, TTA, side model or shared model code.
    injection='''
# Explicit44 ablation of one optional postprocessor, before the R6 controller starts.
DISABLED_R6_POSTPROCESSOR = __FLAG__
ORIGINAL_R6_WORKER_ARGS = list(WORKER_ARGS)
def remove_one_option(arguments, option):
    args=list(arguments)
    if args.count(option)!=1:
        raise RuntimeError('Expected exactly one deployed option to ablate: '+option)
    i=args.index(option)
    if i+1>=len(args) or args[i+1].startswith('--'):
        raise RuntimeError('Missing value for ablated option '+option)
    return args[:i]+args[i+2:]
WORKER_ARGS=remove_one_option(WORKER_ARGS,DISABLED_R6_POSTPROCESSOR)
postprocessor_audit={'experiment':__TAG__,'disabled':DISABLED_R6_POSTPROCESSOR,
                    'target':__TARGET__,'original_worker_args':ORIGINAL_R6_WORKER_ARGS,
                    'actual_worker_args':WORKER_ARGS,'model_and_preprocessing_changed':False}
(WORK/'44_postprocessor_audit.json').write_text(json.dumps(postprocessor_audit,indent=2))
print('44_POSTPROCESSOR_ABLATION',json.dumps(postprocessor_audit),flush=True)
'''.replace('__FLAG__',repr(flag)).replace('__TAG__',repr(tag)).replace('__TARGET__',repr(target))
    modified=original.replace(guard,injection+'\n'+guard)
    compile(modified,'<'+tag+' isolated anchor>','exec')
    lines=parent.splitlines(keepends=True)
    lines[anchor_node.lineno-1:anchor_node.end_lineno]=['ANCHOR_SCRIPT = '+repr(modified)+'\n']
    source=''.join(lines)
    assert '40B_anchor33_S6_probability_0.1' in source
    source=source.replace('40B_anchor33_S6_probability_0.1',tag+'_40B_without_'+('fluid' if tag=='44A' else 'state')+'_stacker')
    source=source.replace("'note':'One-factor probe versus39; no fitted weights or assumed Private gain.'",
                          "'note':"+repr('One optional R6 postprocessor removed versus40B; not an unchanged33 predictor; gain unverified.'))
    marker="anchor_frame=pd.read_csv(WORK/'39_anchor33.csv',dtype={'StudyInstanceUID':str})"
    assert source.count(marker)==1
    source=source.replace(marker,"""ablation=json.loads((WORK/'44_postprocessor_audit.json').read_text())
    if ablation.get('experiment')!=__TAG__ or ablation.get('disabled')!=__FLAG__:
        raise RuntimeError('Missing/mismatched44 postprocessor ablation evidence')
    REPORT['r6_postprocessor_ablation']=ablation
    REPORT['anchor_prediction_is_modified']=True
    """.replace('__TAG__',repr(tag)).replace('__FLAG__',repr(flag))+marker)
    source=source.replace("'fusion':'0.90 * unchanged33_rank_blend + 0.10 * probability(S6)'",
                          "'fusion':"+repr('0.90 * modified33 rank blend (R6 '+flag+' disabled) + 0.10 * probability(S6)'))
    source=source.replace("print('40B COMPLETE'",f"print('{tag} COMPLETE'")
    compile(source,'<'+tag+' parent>','exec')
    intro=f'''# {tag} — 40B with R6 {target} postprocessor disabled

**Creates submission.csv; new unscored experiment.** Baseline40B has user-confirmed Public0.958. This changes **only the optional R6 `{flag}`**, exposing the deployed model's raw {target} probability instead of that postprocessor's output. All model weights, physical crop, slice selection, R6 mirrorTTA, side model and other postprocessors remain unchanged. Fusion follows40B:90% modified33 rank blend +10% raw S6 probability. Because R6 postprocessing changes, the33-derived prediction here is NOT unchanged33.

Rationale: experiment30 examined raw versus deployed outputs on58reused gold. The apparent gains were tiny and their paired intervals contained zero. Gold is not independent validation. This prospective ablation asks whether this specific postprocessor helps or harms Public; it is NOT selected per-case or fitted on gold. No new training or crop tweaks. Do not assume the stacker was fitted on gold: its fitting provenance remains uncertain.

Attach **exactly the same FIVE input datasets/versions as successful40B/39**, including the R6w1 bundle, both publication Raptor checkpoints, S6v5 and Heliosli code/anatomyv1. No old Output needed. **T4x2, Internet Off, Save & Run All.** Use a separate Kaggle notebook for44A/B. They are independent probes;44A removes the Effusion fluid stacker,44B removes the Baker's state stacker, never both in either file.

Artifact paths intentionally remain `rsna_39/39_run_audit.json` and `39_anchor33.csv` to retain tested infrastructure. Require experiment starting{tag}, status COMPLETE, `r6_postprocessor_ablation.disabled` equal`{flag}`, inspect warnings, then submit once. `44_postprocessor_audit.json` records the exact worker arguments. This file and audit must agree. Send score and full Output ZIP.

No GPU/hidden-run verification is claimed; local CPU tests cover argument removal, source preservation and controller paths only. Eight-hour cap is a timeout, not an ETA.40A's hidden exception remains unresolved; unchanged infrastructure can still fail. Do not resubmit an unchanged failure. Keep33/39/40B protected, and reserve the THIRD remaining slot until results return. No guarantee of0.96+ or PrivateTop10.

Third-party credits, code/weight licences and external-data provenance obligations remain inherited from40B. This is not an eligibility determination.
'''
    nb=copy.deepcopy(base)
    nb['cells'][0]['id']=tag+'-intro'
    nb['cells'][0]['source']=intro.splitlines(keepends=True)
    nb['cells'][1]['id']=tag+'-controller'
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    name=tag+'_40B_No'+('EffusionFluid' if tag=='44A' else 'BakerState')+'Stacker_Submit.ipynb'
    path=ROOT/name
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    manifest[tag]={'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'disabled_flag':flag,'target':target,'modified_anchor_sha256':hashlib.sha256(modified.encode()).hexdigest(),
                  'gpu_verified':False,'reference':'40B','combined_ablation':False}
(ROOT/'44_Build_Manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
