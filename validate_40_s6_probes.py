"""Reuse the full validated39 controller harness for both one-factor40 variants."""
import ast, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
base=json.loads((ROOT/'39_Anchor0958_Plus_S6_Fixed10_Rank_Submit.ipynb').read_text(encoding='utf-8'))
def literals(nb):
    t=ast.parse(''.join(nb['cells'][1]['source']))
    return {n.targets[0].id:ast.literal_eval(n.value) for n in t.body
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ('S6_COMPONENT','S6_SCRIPT','ANCHOR_SCRIPT','VISIBLE_REF')}
base_lit=literals(base)
manifest=json.loads((ROOT/'40_Build_Manifest.json').read_text())
harness=(ROOT/'validate_39_submit.py').read_text(encoding='utf-8')
for tag,spec in manifest.items():
    path=ROOT/spec['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
    nb=json.loads(path.read_text(encoding='utf-8'))
    assert literals(nb)==base_lit, 'Inference/model sources or visible references unexpectedly changed'
    assert nb['metadata']['kaggle']['dataSources']==base['metadata']['kaggle']['dataSources']
    test=harness.replace('39_Anchor0958_Plus_S6_Fixed10_Rank_Submit.ipynb',spec['file'])
    test=test.replace("manifest=json.loads((ROOT/'39_Build_Manifest.json').read_text())",
                      'manifest='+repr({'notebook_sha256':spec['sha256']}))
    test=test.replace('39_Local_Validation.json',tag+'_Local_Validation.json')
    if tag=='40A':
        test=test.replace("expected=.9*a[labels].to_numpy()+.1*b.set_index('StudyInstanceUID').loc[ids,labels].rank(method='average',pct=True).to_numpy()",
                          "expected=.8*a[labels].to_numpy()+.2*b.set_index('StudyInstanceUID').loc[ids,labels].rank(method='average',pct=True).to_numpy()")
    else:
        test=test.replace("expected=.9*a[labels].to_numpy()+.1*b.set_index('StudyInstanceUID').loc[ids,labels].rank(method='average',pct=True).to_numpy()",
                          "expected=.9*a[labels].to_numpy()+.1*b.set_index('StudyInstanceUID').loc[ids,labels].to_numpy()")
        test=test.replace('rank fusion with ties and reordered IDs','probability fusion with reordered IDs')
    print('Testing',tag,flush=True)
    exec(compile(test,'<'+tag+' full inherited harness>','exec'),{'__file__':str(ROOT/'validate_39_submit.py'),'__name__':'__main__'})
print('PASS both40 notebooks: inference source identities unchanged; full CPU controller tests; GPU/hidden runs NOT tested.')
