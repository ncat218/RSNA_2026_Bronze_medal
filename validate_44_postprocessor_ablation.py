"""CPU tests: exact optional-flag removal and full parent success/failure paths."""
import ast, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'44_Build_Manifest.json').read_text())
base=json.loads((ROOT/'40B_Anchor0958_Plus_S6_Fixed10_Probability_Submit.ipynb').read_text(encoding='utf-8'))
base_source=''.join(base['cells'][1]['source'])
def literal(source,name):
    return ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n,ast.Assign)
                                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
original_anchor=literal(base_source,'ANCHOR_SCRIPT')
harness=(ROOT/'validate_39_submit.py').read_text(encoding='utf-8')
for tag,spec in manifest.items():
    path=ROOT/spec['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
    nb=json.loads(path.read_text(encoding='utf-8'))
    source=''.join(nb['cells'][1]['source'])
    compile(source,'<'+tag+'>','exec')
    assert literal(source,'S6_COMPONENT')==literal(base_source,'S6_COMPONENT')
    assert literal(source,'S6_SCRIPT')==literal(base_source,'S6_SCRIPT')
    modified=literal(source,'ANCHOR_SCRIPT')
    assert hashlib.sha256(modified.encode()).hexdigest()==spec['modified_anchor_sha256']
    # The original anchor is preserved by removing only the explicit injected block.
    start=modified.index('# Explicit44 ablation of one optional postprocessor')
    end=modified.index('deadline = T0 + DEADLINE_HOURS * 3600',start)
    restored=modified[:start]+modified[end:]
    # Insertion adds whitespace only before the original controller statement.
    assert ast.dump(ast.parse(restored),include_attributes=False)==ast.dump(ast.parse(original_anchor),include_attributes=False)
    helper=next(n for n in ast.parse(modified).body if isinstance(n,ast.FunctionDef) and n.name=='remove_one_option')
    scope={}
    exec(compile(ast.Module(body=[helper],type_ignores=[]),'<44 actual option helper>','exec'),scope)
    args=['--flip-tta','--fluid-stacker','fluid.json','--state-stacker','state.json','--side-model','side.pt']
    result=scope['remove_one_option'](args,spec['disabled_flag'])
    assert spec['disabled_flag'] not in result
    other='--state-stacker' if tag=='44A' else '--fluid-stacker'
    assert other in result and '--flip-tta' in result and '--side-model' in result
    assert len(result)==len(args)-2 and args==['--flip-tta','--fluid-stacker','fluid.json','--state-stacker','state.json','--side-model','side.pt']
    for bad in (['--flip-tta'],[spec['disabled_flag']],[spec['disabled_flag'],'a',spec['disabled_flag'],'b']):
        try:scope['remove_one_option'](bad,spec['disabled_flag'])
        except RuntimeError:pass
        else:raise AssertionError('Ambiguous missing/duplicate option accepted')
    test=harness.replace('39_Anchor0958_Plus_S6_Fixed10_Rank_Submit.ipynb',spec['file'])
    test=test.replace("manifest=json.loads((ROOT/'39_Build_Manifest.json').read_text())",
                      'manifest='+repr({'notebook_sha256':spec['sha256']}))
    test=test.replace('39_Local_Validation.json',tag+'_Local_Validation.json')
    test=test.replace("expected=.9*a[labels].to_numpy()+.1*b.set_index('StudyInstanceUID').loc[ids,labels].rank(method='average',pct=True).to_numpy()",
                      "expected=.9*a[labels].to_numpy()+.1*b.set_index('StudyInstanceUID').loc[ids,labels].to_numpy()")
    marker="frame.to_csv(WORK/'39_anchor33.csv',index=False)"
    assert test.count(marker)==1
    test=test.replace(marker,"(WORK/'44_postprocessor_audit.json').write_text(json.dumps("+repr({'experiment':tag,'disabled':spec['disabled_flag']})+"))\n        "+marker)
    test=test.replace('rank fusion with ties and reordered IDs','40B probability fusion with reordered IDs')
    print('Testing',tag,flush=True)
    exec(compile(test,'<'+tag+' inherited full parent harness>','exec'),{'__file__':str(ROOT/'validate_39_submit.py'),'__name__':'__main__'})
    report_path=ROOT/(tag+'_Local_Validation.json')
    report=json.loads(report_path.read_text())
    report['tests']+=['only injected worker-argument filter differs from original40B anchor adapter',
                       'S6 stage/component source byte-identical to40B',
                       'exactly one optional flag+path removed; other stacker and mirror retained',
                       'missing/duplicate option rejected']
    report['gpu_or_actual_stacker_inference_tested']=False
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
print('PASS44A/B CPU tests. No actual GPU/hidden inference verified.')
